#!/usr/bin/env python3
"""
Find outlier predictions where normalized distance > 100% of image diagonal.
"""

import json
import math
import os
from PIL import Image

# Configuration
JSON_FILES = {
    "baseline": "/Users/vincent/Desktop/test/GTA1/run_screenspot_pro/results/gta1_baseline_100.json",
    "distance_reward": "/Users/vincent/Desktop/test/GTA1/run_screenspot_pro/results/gta1_distance_reward_100.json",
}
IMAGE_DIR = "/Users/vincent/Desktop/test/GTA1/screenspot_pro_mini_evalset"

# Threshold for outlier (% of image diagonal)
OUTLIER_THRESHOLD = 100  # 100% of diagonal


def get_image_diagonal(img_path):
    """Get image dimensions and diagonal."""
    parts = img_path.split("/")
    rel_path = "/".join(parts[-2:])
    local_path = os.path.join(IMAGE_DIR, rel_path)
    if os.path.exists(local_path):
        with Image.open(local_path) as img:
            w, h = img.size
            return math.sqrt(w**2 + h**2), w, h
    return None, None, None


def distance_to_bbox_edge(pred, bbox):
    """Calculate minimum distance from prediction to bbox edge."""
    x1, y1, x2, y2 = bbox
    px, py = pred
    if x1 <= px <= x2 and y1 <= py <= y2:
        return 0.0
    closest_x = max(x1, min(px, x2))
    closest_y = max(y1, min(py, y2))
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def find_outliers(json_path, model_name):
    """Find outliers in a JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    
    outliers = []
    for entry in data["details"]:
        if entry["correctness"] == "wrong":
            pred = entry["pred"]
            bbox = entry["bbox"]
            img_path = entry["img_path"]
            dist = distance_to_bbox_edge(pred, bbox)
            diagonal, w, h = get_image_diagonal(img_path)
            
            if diagonal:
                norm_dist = (dist / diagonal) * 100
                if norm_dist > OUTLIER_THRESHOLD:
                    outliers.append({
                        "raw_distance": dist,
                        "normalized_distance": norm_dist,
                        "image_size": (w, h),
                        "diagonal": diagonal,
                        "prompt": entry["prompt_to_evaluate"],
                        "image": img_path.split("/")[-1],
                        "pred": pred,
                        "bbox": bbox,
                        "raw_response": entry.get("raw_response", ""),
                    })
    
    return outliers


def main():
    print(f"Finding outliers with normalized distance > {OUTLIER_THRESHOLD}% of diagonal\n")
    
    for model_name, json_path in JSON_FILES.items():
        print(f"{'='*80}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*80}")
        
        outliers = find_outliers(json_path, model_name)
        
        if outliers:
            for i, o in enumerate(outliers, 1):
                print(f"\nOutlier #{i}:")
                print(f"  Raw Distance: {o['raw_distance']:.1f}px")
                print(f"  Image size: {o['image_size'][0]}x{o['image_size'][1]}, Diagonal: {o['diagonal']:.1f}px")
                print(f"  NORMALIZED Distance: {o['normalized_distance']:.1f}% of diagonal")
                print(f"  Prompt: {o['prompt']}")
                print(f"  Image: {o['image']}")
                print(f"  Pred: ({o['pred'][0]:.0f}, {o['pred'][1]:.0f})")
                print(f"  Bbox: {o['bbox']}")
                print(f"  Raw response: {o['raw_response']}")
        else:
            print("\nNo outliers found!")
        
        print()


if __name__ == "__main__":
    main()

