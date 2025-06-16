# Use a PyTorch base image with CUDA support suitable for the model
# Check RunPod documentation for recommended base images if any,
# otherwise, a standard PyTorch image should work.
# Example: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 as a base for torch
# For LTX-Video, Python 3.10 is recommended.
# PyTorch 2.1.0+ is listed in pyproject.toml.
# Let's try to find a PyTorch image that matches these.
# runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04 seems like a good candidate from RunPod's examples.
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install essential build tools and ffmpeg (for imageio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for models and download them using a Python script
# This ensures precise placement and avoids issues with CLI creating subdirectories.
RUN mkdir -p /app/MODEL_DIR && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltxv-13b-0.9.7-distilled.safetensors', local_dir='/app/MODEL_DIR', local_dir_use_symlinks=False, repo_type='model')" && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Lightricks/LTX-Video', filename='ltxv-spatial-upscaler-0.9.7.safetensors', local_dir='/app/MODEL_DIR', local_dir_use_symlinks=False, repo_type='model')"

# Download the PixArt-alpha text encoder model components
# These are specified by text_encoder_model_name_or_path in the LTX config.
# We use snapshot_download to get the entire model repo, ensuring all necessary files
# (like config.json, model weights, tokenizer files, and subfolders) are present.
# The target directory name is sanitized to be a valid directory name.
RUN python -c "from huggingface_hub import snapshot_download; import os; \
    repo_id='PixArt-alpha/PixArt-XL-2-1024-MS'; \
    sanitized_repo_id = repo_id.replace('/', '_'); \
    local_dir=os.path.join('/app/text_encoder_models_cache', sanitized_repo_id); \
    os.makedirs(local_dir, exist_ok=True); \
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, \
                      allow_patterns=['*.json', '*.safetensors', '*.bin', '*.md', 'spiece.model'])"

# Copy the rest of the application code
# This includes inference.py, the ltx_video module, configs, etc.
COPY . .

# Ensure the handler script is executable (though Python scripts usually don't need +x if called with python)
# RUN chmod +x /app/handler.py

# Set the default command to run the RunPod serverless worker
# The runpod library's serverless.start will be initiated by handler.py
CMD ["python", "-u", "/app/handler.py"]
