#!/usr/bin/env python3
"""
Download images from the ScreenSpot-Pro HuggingFace dataset based on paths in the JSON file.
"""

import json
import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
JSON_PATH = "/Users/vincent/Desktop/test/GTA1/run_screenspot_pro/results/base_model_3b.json"
OUTPUT_DIR = "/Users/vincent/Desktop/test/GTA1/screenspot_pro_mini_evalset"
HF_BASE_URL = "https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/resolve/main/images"

def extract_relative_path(img_path: str) -> str:
    """Extract the last 2 parts of the path (folder/filename)."""
    parts = img_path.split("/")
    # Get last 2 parts: folder name and filename
    return "/".join(parts[-2:])

def download_image(relative_path: str, output_dir: str) -> tuple[str, bool, str]:
    """
    Download a single image from HuggingFace.
    Returns: (relative_path, success, message)
    """
    # URL encode spaces in filenames
    url_path = relative_path.replace(" ", "%20")
    url = f"{HF_BASE_URL}/{url_path}"
    
    # Create subdirectory if needed
    local_path = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Skip if already exists
    if os.path.exists(local_path):
        return (relative_path, True, "Already exists")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(local_path, "wb") as f:
            f.write(response.content)
        
        return (relative_path, True, "Downloaded")
    except requests.exceptions.RequestException as e:
        return (relative_path, False, str(e))

def main():
    # Load JSON
    print(f"Loading JSON from {JSON_PATH}...")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    
    # Extract unique relative paths
    details = data.get("details", [])
    relative_paths = set()
    for item in details:
        img_path = item.get("img_path", "")
        if img_path:
            rel_path = extract_relative_path(img_path)
            relative_paths.add(rel_path)
    
    print(f"Found {len(relative_paths)} unique images to download")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Download images with progress bar
    success_count = 0
    fail_count = 0
    failed_images = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(download_image, path, OUTPUT_DIR): path 
            for path in relative_paths
        }
        
        with tqdm(total=len(relative_paths), desc="Downloading images") as pbar:
            for future in as_completed(futures):
                rel_path, success, message = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    failed_images.append((rel_path, message))
                pbar.update(1)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    if failed_images:
        print(f"\nFailed downloads:")
        for path, error in failed_images:
            print(f"  - {path}: {error}")

if __name__ == "__main__":
    main()

