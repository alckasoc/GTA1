#!/usr/bin/env python3
"""
Annotate images with ground truth bounding boxes and predicted coordinates.
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Configuration
JSON_PATH = "/Users/vincent/Desktop/test/GTA1/run_screenspot_pro/results/gta1_distance_reward_100.json"
IMAGE_DIR = "/Users/vincent/Desktop/test/GTA1/screenspot_pro_mini_evalset"
OUTPUT_DIR = "/Users/vincent/Desktop/test/GTA1/screenspot_pro_mini_annotated_gta1_distance_reward_100"

# Set to True to only annotate correct predictions, False to annotate all
CORRECT_ONLY = True

# Colors
GT_COLOR = (0, 255, 0)  # Green for ground truth bbox
PRED_COLOR_CORRECT = (0, 255, 0)  # Green for correct prediction
PRED_COLOR_WRONG = (255, 0, 0)  # Red for wrong prediction
BBOX_WIDTH = 4
PRED_RADIUS = 15


def extract_relative_path(img_path: str) -> str:
    """Extract the last 2 parts of the path (folder/filename)."""
    parts = img_path.split("/")
    return "/".join(parts[-2:])


def draw_bbox(draw: ImageDraw.Draw, bbox: list, color: tuple, width: int = 3):
    """Draw a bounding box on the image."""
    x1, y1, x2, y2 = bbox
    # Draw rectangle
    for i in range(width):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)


def draw_point(draw: ImageDraw.Draw, point: list, color: tuple, radius: int = 10):
    """Draw a point (circle with crosshair) on the image."""
    x, y = int(point[0]), int(point[1])
    # Draw filled circle
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=color)
    # Draw crosshair
    crosshair_len = radius + 10
    draw.line([x - crosshair_len, y, x + crosshair_len, y], fill=color, width=3)
    draw.line([x, y - crosshair_len, x, y + crosshair_len], fill=color, width=3)


def draw_label(draw: ImageDraw.Draw, text: str, position: tuple, color: tuple, bg_color: tuple = (0, 0, 0)):
    """Draw a label with background."""
    x, y = position
    # Get text size (approximate)
    text_width = len(text) * 12
    text_height = 20
    padding = 5
    
    # Draw background
    draw.rectangle(
        [x, y, x + text_width + padding * 2, y + text_height + padding * 2],
        fill=bg_color
    )
    # Draw text
    draw.text((x + padding, y + padding), text, fill=color)


def annotate_image(entry: dict, image_dir: str, output_dir: str, correct_only: bool = False) -> tuple[bool, str]:
    """
    Annotate a single image with bbox and prediction.
    Returns: (success, message)
    """
    img_path = entry.get("img_path", "")
    bbox = entry.get("bbox", [])
    pred = entry.get("pred", [])
    correctness = entry.get("correctness", "")
    prompt = entry.get("prompt_to_evaluate", "")
    
    # Skip entries that are not correct (if flag is set)
    if correct_only and correctness != "correct":
        return False, "Skipped (not correct)"
    
    if not img_path or not bbox or not pred:
        return False, "Missing required fields"
    
    # Get relative path and local image path
    rel_path = extract_relative_path(img_path)
    local_img_path = os.path.join(image_dir, rel_path)
    
    if not os.path.exists(local_img_path):
        return False, f"Image not found: {local_img_path}"
    
    try:
        # Load image
        img = Image.open(local_img_path)
        draw = ImageDraw.Draw(img)
        
        # Determine prediction color based on correctness
        pred_color = PRED_COLOR_CORRECT if correctness == "correct" else PRED_COLOR_WRONG
        
        # Draw ground truth bbox (green)
        draw_bbox(draw, bbox, GT_COLOR, BBOX_WIDTH)
        
        # Draw predicted point
        draw_point(draw, pred, pred_color, PRED_RADIUS)
        
        # Add labels
        # GT label near bbox
        draw_label(draw, "GT", (bbox[0], bbox[1] - 30), GT_COLOR)
        
        # Pred label near prediction point
        pred_label = f"Pred ({correctness})"
        draw_label(draw, pred_label, (int(pred[0]) + 20, int(pred[1]) - 10), pred_color)
        
        # Add prompt at top of image
        draw_label(draw, f"Prompt: {prompt}", (10, 10), (255, 255, 255), (50, 50, 50))
        
        # Create output path maintaining folder structure
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save annotated image
        img.save(output_path)
        
        return True, "Annotated"
    
    except Exception as e:
        return False, str(e)


def main():
    # Load JSON
    print(f"Loading JSON from {JSON_PATH}...")
    print(f"CORRECT_ONLY: {CORRECT_ONLY}")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    
    details = data.get("details", [])
    print(f"Found {len(details)} entries to process")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each entry
    success_count = 0
    skipped_count = 0
    error_count = 0
    error_entries = []
    
    for entry in tqdm(details, desc="Annotating images"):
        success, message = annotate_image(entry, IMAGE_DIR, OUTPUT_DIR, correct_only=CORRECT_ONLY)
        if success:
            success_count += 1
        elif message == "Skipped (not correct)":
            skipped_count += 1
        else:
            error_count += 1
            error_entries.append((extract_relative_path(entry.get("img_path", "")), message))
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Annotation complete!")
    print(f"  Annotated: {success_count}")
    if CORRECT_ONLY:
        print(f"  Skipped (not correct): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    if error_entries:
        print(f"\nError entries:")
        for path, error in error_entries[:10]:  # Show first 10
            print(f"  - {path}: {error}")
        if len(error_entries) > 10:
            print(f"  ... and {len(error_entries) - 10} more")


if __name__ == "__main__":
    main()

