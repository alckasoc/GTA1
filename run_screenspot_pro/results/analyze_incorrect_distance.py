#!/usr/bin/env python3
"""
Analyze incorrect predictions - compare distance from prediction to bbox
between baseline and distance reward models.
Distances are normalized by image diagonal.
"""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os

# Files to analyze
JSON_FILES = {
    "baseline": "/Users/vincent/Desktop/test/GTA1/run_screenspot_pro/results/gta1_baseline_100.json",
    "distance_reward": "/Users/vincent/Desktop/test/GTA1/run_screenspot_pro/results/gta1_distance_reward_100.json",
}

IMAGE_DIR = "/Users/vincent/Desktop/test/GTA1/screenspot_pro_mini_evalset"
OUTPUT_DIR = "/Users/vincent/Desktop/test/GTA1/run_screenspot_pro/results"

# Cache for image dimensions
_image_dims_cache = {}


def get_image_dimensions(img_path: str) -> tuple:
    """Get image width and height, with caching."""
    if img_path in _image_dims_cache:
        return _image_dims_cache[img_path]
    
    # Extract relative path (last 2 parts)
    parts = img_path.split("/")
    rel_path = "/".join(parts[-2:])
    local_path = os.path.join(IMAGE_DIR, rel_path)
    
    if os.path.exists(local_path):
        with Image.open(local_path) as img:
            dims = img.size  # (width, height)
            _image_dims_cache[img_path] = dims
            return dims
    
    # Fallback: return None
    return None


def get_image_diagonal(img_path: str) -> float:
    """Get image diagonal for normalization."""
    dims = get_image_dimensions(img_path)
    if dims:
        return math.sqrt(dims[0]**2 + dims[1]**2)
    return None


def distance_to_bbox_center(pred: list, bbox: list) -> float:
    """Calculate distance from prediction to bbox center."""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return math.sqrt((pred[0] - center_x)**2 + (pred[1] - center_y)**2)


def distance_to_bbox_edge(pred: list, bbox: list) -> float:
    """
    Calculate minimum distance from prediction to bbox.
    If inside bbox, returns 0.
    If outside, returns distance to nearest edge.
    """
    x1, y1, x2, y2 = bbox
    px, py = pred
    
    # Check if inside bbox
    if x1 <= px <= x2 and y1 <= py <= y2:
        return 0.0
    
    # Clamp to bbox and calculate distance
    closest_x = max(x1, min(px, x2))
    closest_y = max(y1, min(py, y2))
    
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def analyze_json(json_path: str) -> dict:
    """Analyze incorrect predictions in a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    details = data.get("details", [])
    
    # Filter for incorrect only
    incorrect = [e for e in details if e.get("correctness") == "wrong"]
    
    distances_to_center = []
    distances_to_edge = []
    normalized_distances_to_center = []  # As % of image diagonal
    normalized_distances_to_edge = []    # As % of image diagonal
    
    for entry in incorrect:
        pred = entry.get("pred", [])
        bbox = entry.get("bbox", [])
        img_path = entry.get("img_path", "")
        
        if pred and bbox:
            dist_center = distance_to_bbox_center(pred, bbox)
            dist_edge = distance_to_bbox_edge(pred, bbox)
            distances_to_center.append(dist_center)
            distances_to_edge.append(dist_edge)
            
            # Get image diagonal for normalization
            diagonal = get_image_diagonal(img_path)
            if diagonal:
                normalized_distances_to_center.append((dist_center / diagonal) * 100)
                normalized_distances_to_edge.append((dist_edge / diagonal) * 100)
    
    return {
        "total": len(details),
        "num_incorrect": len(incorrect),
        "distances_to_center": distances_to_center,
        "distances_to_edge": distances_to_edge,
        "normalized_distances_to_center": normalized_distances_to_center,
        "normalized_distances_to_edge": normalized_distances_to_edge,
    }


def main():
    results = {}
    
    print("Analyzing incorrect predictions (with normalization by image diagonal)...\n")
    
    for name, json_path in JSON_FILES.items():
        print(f"{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")
        
        stats = analyze_json(json_path)
        results[name] = stats
        
        print(f"Total entries: {stats['total']}")
        print(f"Incorrect predictions: {stats['num_incorrect']}")
        
        if stats['normalized_distances_to_edge']:
            print(f"\nNORMALIZED Distance to bbox EDGE (% of image diagonal):")
            print(f"  Mean: {np.mean(stats['normalized_distances_to_edge']):.2f}%")
            print(f"  Median: {np.median(stats['normalized_distances_to_edge']):.2f}%")
            print(f"  Std: {np.std(stats['normalized_distances_to_edge']):.2f}%")
            print(f"  Min: {np.min(stats['normalized_distances_to_edge']):.2f}%")
            print(f"  Max: {np.max(stats['normalized_distances_to_edge']):.2f}%")
            
            print(f"\nNORMALIZED Distance to bbox CENTER (% of image diagonal):")
            print(f"  Mean: {np.mean(stats['normalized_distances_to_center']):.2f}%")
            print(f"  Median: {np.median(stats['normalized_distances_to_center']):.2f}%")
            print(f"  Std: {np.std(stats['normalized_distances_to_center']):.2f}%")
            print(f"  Min: {np.min(stats['normalized_distances_to_center']):.2f}%")
            print(f"  Max: {np.max(stats['normalized_distances_to_center']):.2f}%")
        
        print()
    
    # Create comparison plots - NORMALIZED
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors
    colors = {"baseline": "#3498db", "distance_reward": "#e74c3c"}
    
    # Plot 1: Histogram of NORMALIZED distance to edge
    ax1 = axes[0, 0]
    for name, stats in results.items():
        ax1.hist(stats['normalized_distances_to_edge'], bins=30, alpha=0.6, 
                label=f"{name} (n={stats['num_incorrect']})", color=colors[name])
    ax1.set_xlabel('Normalized Distance to Bbox Edge (% of diagonal)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Normalized Distance to Bbox Edge')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of NORMALIZED distance to center
    ax2 = axes[0, 1]
    for name, stats in results.items():
        ax2.hist(stats['normalized_distances_to_center'], bins=30, alpha=0.6,
                label=f"{name} (n={stats['num_incorrect']})", color=colors[name])
    ax2.set_xlabel('Normalized Distance to Bbox Center (% of diagonal)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Normalized Distance to Bbox Center')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: CDF of NORMALIZED distance to edge
    ax3 = axes[1, 0]
    for name, stats in results.items():
        sorted_dist = np.sort(stats['normalized_distances_to_edge'])
        cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
        ax3.plot(sorted_dist, cdf, label=f"{name}", color=colors[name], linewidth=2)
    ax3.set_xlabel('Normalized Distance to Bbox Edge (% of diagonal)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('CDF of Normalized Distance to Bbox Edge')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    
    # Plot 4: Box plot comparison - NORMALIZED
    ax4 = axes[1, 1]
    data_edge = [results['baseline']['normalized_distances_to_edge'], 
                 results['distance_reward']['normalized_distances_to_edge']]
    bp = ax4.boxplot(data_edge, labels=['Baseline', 'Distance Reward'], patch_artist=True)
    bp['boxes'][0].set_facecolor(colors['baseline'])
    bp['boxes'][1].set_facecolor(colors['distance_reward'])
    ax4.set_ylabel('Normalized Distance to Bbox Edge (% of diagonal)')
    ax4.set_title('Box Plot Comparison (Normalized Distance to Edge)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = Path(OUTPUT_DIR) / "incorrect_distance_analysis_normalized.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    
    # Also save a summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON (NORMALIZED)")
    print("="*60)
    print(f"\n{'Metric':<40} {'Baseline':>12} {'Dist Reward':>12} {'Diff':>10}")
    print("-"*75)
    
    b_edge = results['baseline']['normalized_distances_to_edge']
    d_edge = results['distance_reward']['normalized_distances_to_edge']
    
    metrics = [
        ("Mean normalized dist to edge (%)", np.mean(b_edge), np.mean(d_edge)),
        ("Median normalized dist to edge (%)", np.median(b_edge), np.median(d_edge)),
        ("Std normalized dist to edge (%)", np.std(b_edge), np.std(d_edge)),
        ("% within 5% of diagonal", 
         sum(1 for x in b_edge if x < 5) / len(b_edge) * 100,
         sum(1 for x in d_edge if x < 5) / len(d_edge) * 100),
        ("% within 10% of diagonal",
         sum(1 for x in b_edge if x < 10) / len(b_edge) * 100,
         sum(1 for x in d_edge if x < 10) / len(d_edge) * 100),
        ("% within 20% of diagonal",
         sum(1 for x in b_edge if x < 20) / len(b_edge) * 100,
         sum(1 for x in d_edge if x < 20) / len(d_edge) * 100),
    ]
    
    for metric_name, b_val, d_val in metrics:
        diff = d_val - b_val
        sign = "+" if diff > 0 else ""
        print(f"{metric_name:<40} {b_val:>12.2f} {d_val:>12.2f} {sign}{diff:>9.2f}")
    
    plt.show()


if __name__ == "__main__":
    main()

