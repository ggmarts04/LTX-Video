import os
import json
import tempfile
import uuid
import runpod
import subprocess # For calling inference.py if needed, though direct import is better
from pathlib import Path
import random

# Assuming inference.py is in the same directory and has been modified
# to accept URLs and has the main infer logic.
from inference import infer, get_device, load_image_to_tensor_with_resize_and_crop, download_image_from_url
from PIL import Image # For saving downloaded images if needed locally before passing path

# Configuration
DEFAULT_MODEL_CONFIG = "configs/ltxv-13b-0.9.7-distilled.yaml"
DEFAULT_HEIGHT = 704
DEFAULT_WIDTH = 1216
DEFAULT_NUM_FRAMES = 121
DEFAULT_FRAME_RATE = 30

def handler(job):
    job_input = job.get('input', {})

    # --- Input Validation and Defaults ---
    prompt = job_input.get('prompt')
    if not prompt:
        return {"error": "Prompt is a required input."}

    negative_prompt = job_input.get('negative_prompt', "worst quality, inconsistent motion, blurry, jittery, distorted")
    height = job_input.get('height', DEFAULT_HEIGHT)
    width = job_input.get('width', DEFAULT_WIDTH)
    num_frames = job_input.get('num_frames', DEFAULT_NUM_FRAMES)
    frame_rate = job_input.get('frame_rate', DEFAULT_FRAME_RATE)
    seed = job_input.get('seed', random.randint(0, 2**32 - 1))

    conditioning_images_input = job_input.get('conditioning_images', [])
    if not conditioning_images_input:
        return {"error": "At least one conditioning image (first frame) is required for image-to-video."}

    conditioning_media_paths = []
    conditioning_start_frames = []
    conditioning_strengths = []

    # --- Prepare output directory ---
    # Create a unique temporary directory for each job's outputs
    # RunPod typically provides an output path or expects files in a specific location.
    # For serverless, often files are written to /tmp or a designated output dir.
    # Let's use a unique output directory within /tmp for generated videos.
    # The `infer` script saves files in `outputs/DATE/` by default.
    # We can override this or let it be, then move the file.
    # For simplicity, let's create a unique output dir and pass it to `infer`.

    # Generate a unique name for the output directory for this job
    # tempfile.mkdtemp() creates a secure temporary directory.
    # However, `infer` creates its own subdirectories.
    # Let's create a base output directory for this job.
    job_output_dir = Path(tempfile.mkdtemp(prefix="ltx_video_output_"))

    # --- Process Conditioning Images ---
    # Download images and store them temporarily if `infer` can't take URLs directly
    # (even though we modified it, this provides a fallback or alternative)
    # For now, assuming `infer` and its sub-functions correctly handle URLs.
    for image_info in conditioning_images_input:
        url = image_info.get('url')
        start_frame = image_info.get('start_frame')
        strength = image_info.get('strength', 1.0)

        if not url or start_frame is None:
            return {"error": "Each conditioning image must have a 'url' and 'start_frame'."}

        # The modified inference.py should handle URLs directly.
        conditioning_media_paths.append(url)
        conditioning_start_frames.append(int(start_frame))
        conditioning_strengths.append(float(strength))

    if not conditioning_media_paths:
         return {"error": "No valid conditioning images provided or processed."}

    # --- Run Inference ---
    try:
        # Set device (cuda if available, else cpu)
        device = get_device()

        # The infer script saves outputs to a subdirectory.
        # We need to capture the path of the generated video.
        # The `infer` function in `inference.py` already creates unique filenames.
        # We'll pass our job_output_dir as the base output_path.

        # The `infer` function will create something like:
        # `job_output_dir/video_output_0_PROMPT_SEED_RESOLUTION_INDEX.mp4`
        # We need to find this file after `infer` runs.

        print(f"Starting inference with output directory: {job_output_dir}")
        infer(
            output_path=str(job_output_dir), # Pass our unique dir
            seed=int(seed),
            pipeline_config=DEFAULT_MODEL_CONFIG,
            image_cond_noise_scale=0.15, # Default from inference.py, or make it an input
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            frame_rate=int(frame_rate),
            prompt=prompt,
            negative_prompt=negative_prompt,
            offload_to_cpu=False, # Adjust based on typical RunPod GPU memory
            input_media_path=None, # We are using conditioning_media_paths
            conditioning_media_paths=conditioning_media_paths,
            conditioning_strengths=conditioning_strengths,
            conditioning_start_frames=conditioning_start_frames,
            device=device
        )

        # --- Locate the output file ---
        # The `infer` script saves files with a unique name.
        # We need to find the generated .mp4 or .png file in job_output_dir.
        # (The `infer` script creates subdirectories like `outputs/DATE/` if `output_path` is None,
        # but if `output_path` is specified, it uses that as the base.)
        # The `get_unique_filename` in `inference.py` saves directly into `output_path` if it's given.

        output_files = list(job_output_dir.glob("*.mp4")) + list(job_output_dir.glob("*.png"))
        if not output_files:
            return {"error": "Inference completed but no output video/image file found."}

        # Assuming one video output for now
        output_file_path = str(output_files[0])

        # RunPod expects a file path for automatic upload for serverless workers,
        # or a dictionary with `s3Path` if we upload manually.
        # Returning the file path should be sufficient if the worker is configured for it.
        # Alternatively, one could return the file content as base64, but videos are large.

        print(f"Inference successful. Output file: {output_file_path}")
        return {"output_video": output_file_path}

    except Exception as e:
        # Log the full error for debugging
        import traceback
        print(f"Error during inference: {e}")
        print(traceback.format_exc())
        return {"error": f"Inference failed: {str(e)}"}

# Start the RunPod worker
if __name__ == "__main__":
    print("Starting RunPod LTX-Video handler...")
    runpod.serverless.start({"handler": handler})
