#!/usr/bin/env python3
"""
Inference script for GTA1 grounding model.
Loads a trained model and runs inference on images with instructions.
"""

import argparse
import os
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from utils.inference_utils import run_single_inference, plot_coordinates_on_image


def run_inference(model_path, image_path, instruction, max_new_tokens=32, 
                  min_pixels=3136, max_pixels=4096*2160, 
                  torch_dtype="bfloat16", attn_implementation="flash_attention_2"):
    """
    Run inference on a single image with an instruction.
    
    Args:
        model_path: Path to the trained model
        image_path: Path to the input image
        instruction: Text instruction describing what to locate
        max_new_tokens: Maximum number of tokens to generate
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing
        torch_dtype: Torch dtype for model loading
        attn_implementation: Attention implementation to use
    
    Returns:
        dict: Results dictionary with predictions and metadata
    """
    # Convert torch_dtype string to torch.dtype if needed
    if isinstance(torch_dtype, str):
        if torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16
    
    # Load model and processor
    print(f"Loading model from {model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map="auto"
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    print("Model loaded successfully!")
    
    # Load image
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert("RGB")
    print(f"Original image size: {image.width}x{image.height}")
    
    # Run inference using shared utility
    results = run_single_inference(
        model=model,
        processor=processor,
        image=image,
        instruction=instruction,
        max_new_tokens=max_new_tokens
    )
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Model output: {results['output_text']}")
    print(f"Predicted coordinates (resized): ({results['pred_x']}, {results['pred_y']})")
    print(f"Predicted coordinates (original): ({results['pred_x_scaled']:.1f}, {results['pred_y_scaled']:.1f})")
    print(f"Scale factors: x={results['scale_x']:.3f}, y={results['scale_y']:.3f}")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with GTA1 grounding model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image file"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Text instruction describing what element to locate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate (default: 32)"
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=3136,
        help="Minimum pixels for image processing (default: 3136)"
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=4096*2160,
        help="Maximum pixels for image processing (default: 4096*2160)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model loading (default: bfloat16)"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation (default: flash_attention_2)"
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default=None,
        help="Optional path to save annotated image with predicted coordinates"
    )
    parser.add_argument(
        "--gt_x",
        type=float,
        default=None,
        help="Optional ground truth x coordinate for visualization"
    )
    parser.add_argument(
        "--gt_y",
        type=float,
        default=None,
        help="Optional ground truth y coordinate for visualization"
    )
    parser.add_argument(
        "--gt_bbox",
        type=str,
        default=None,
        help="Optional ground truth bounding box as 'x1,y1,x2,y2' (e.g., '100,200,300,400')"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    
    # Run inference
    results = run_inference(
        model_path=args.model_path,
        image_path=args.image_path,
        instruction=args.instruction,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation
    )
    
    # Parse bounding box if provided
    gt_bbox = None
    if args.gt_bbox:
        try:
            gt_bbox = [float(x) for x in args.gt_bbox.split(',')]
            if len(gt_bbox) != 4:
                raise ValueError("Bounding box must have 4 values: x1,y1,x2,y2")
        except Exception as e:
            print(f"Warning: Invalid gt_bbox format: {e}. Expected 'x1,y1,x2,y2'")
            gt_bbox = None
    
    # Save annotated image if output path specified
    if args.output_image:
        image = Image.open(args.image_path).convert("RGB")
        plot_coordinates_on_image(
            image=image,
            pred_x=results['pred_x_scaled'],
            pred_y=results['pred_y_scaled'],
            gt_x=args.gt_x,
            gt_y=args.gt_y,
            gt_bbox=gt_bbox,
            save_path=args.output_image,
            show_text=True
        )


if __name__ == "__main__":
    main()

