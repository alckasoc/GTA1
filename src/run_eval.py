#!/usr/bin/env python3
"""Simple evaluation script - run after training to evaluate on a single example."""

import os
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from utils.inference_utils import run_single_inference, plot_coordinates_on_image

def main():
    # Hardcoded paths
    model_path = "/home/ubuntu/GTA1/grounding/test/checkpoint-5"
    image_root = "/home/ubuntu/GTA1/images"
    output_dir = "inference_example"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and processor
    print(f"Loading model from {model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Hardcoded test example
    data_item = {
        "id": 1,
        "image": "dataset/Aria-UI_Data/web/images/screenshot_0b1312fe-a57d-4201-8a49-71f73917ad69_part_3.png",
        "bbox": [714, 694, 731, 724],
        "conversations": [
            {"from": "human", "value": "<image>send your inquiry."},
            {"from": "gpt", "value": "(722, 709)"}
        ]
    }
    
    # Load image
    image_path = os.path.join(image_root, data_item['image'])
    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path} ({image.width}x{image.height})")
    
    # Get instruction
    instruction = data_item['conversations'][0]['value'].replace("<image>", "").strip()
    
    # Run inference
    with torch.no_grad():
        results = run_single_inference(
            model=model,
            processor=processor,
            image=image,
            instruction=instruction,
            max_new_tokens=32
        )
    
    # Extract results
    pred_x_scaled = results['pred_x_scaled']
    pred_y_scaled = results['pred_y_scaled']
    
    # Get ground truth
    gt_bbox = data_item['bbox']
    x0, y0, x1, y1 = gt_bbox
    gt_center_x = (x0 + x1) / 2
    gt_center_y = (y0 + y1) / 2
    
    print(f"  Predicted: ({pred_x_scaled:.1f}, {pred_y_scaled:.1f})")
    print(f"  Ground truth: ({gt_center_x:.1f}, {gt_center_y:.1f})")
    print(f"  Model output: {results['output_text']}")
    
    # Create annotated image
    annotated_image = plot_coordinates_on_image(
        image=image,
        pred_x=pred_x_scaled,
        pred_y=pred_y_scaled,
        gt_x=gt_center_x,
        gt_y=gt_center_y,
        gt_bbox=gt_bbox,
        show_text=True
    )
    
    # Save results
    img_path = os.path.join(output_dir, "example_0_annotated.png")
    annotated_image.save(img_path)
    
    results_data = [{
        "index": 0,
        "annotated_image": img_path,
        "predicted_coordinate": f"({pred_x_scaled:.1f}, {pred_y_scaled:.1f})",
        "ground_truth_coordinate": f"({gt_center_x:.1f}, {gt_center_y:.1f})",
        "ground_truth_bbox": f"[{int(x0)}, {int(y0)}, {int(x1)}, {int(y1)}]"
    }]
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nSaved results to {output_dir}/")

if __name__ == "__main__":
    main()

