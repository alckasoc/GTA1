"""
Shared inference utilities for GTA1 grounding model.
"""

import re
import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoProcessor

SYSTEM_PROMPT = '''
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
'''
SYSTEM_PROMPT = SYSTEM_PROMPT.strip()


def extract_coordinates(raw_string):
    """Extract coordinates from model output."""
    try:
        matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", raw_string)
        return [tuple(map(int, match)) for match in matches][0]
    except:
        return 0, 0


def prepare_image_and_messages(image, instruction, processor, resized_height=None, resized_width=None):
    """
    Prepare image and messages for inference.
    
    Args:
        image: PIL Image (can be original or resized)
        instruction: Text instruction
        processor: AutoProcessor instance
        resized_height: Optional resized height (if None, will compute from image)
        resized_width: Optional resized width (if None, will compute from image)
    
    Returns:
        tuple: (resized_image, system_message, user_message, scale_x, scale_y, original_width, original_height)
    """
    original_width, original_height = image.width, image.height
    
    # Resize image if needed
    if resized_height is None or resized_width is None:
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
            min_pixels=processor.image_processor.min_pixels,
            max_pixels=processor.image_processor.max_pixels,
        )
    
    resized_image = image.resize((resized_width, resized_height))
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    
    # Prepare system and user messages
    system_message = {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT.format(height=resized_height, width=resized_width)}]
    }
    
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": resized_image},
            {"type": "text", "text": instruction}
        ]
    }
    
    return resized_image, system_message, user_message, scale_x, scale_y, original_width, original_height


def generate_prediction(model, processor, system_message, user_message, max_new_tokens=32):
    """
    Generate prediction from model given messages.
    
    Args:
        model: The model to use for generation
        processor: AutoProcessor instance
        system_message: System message dict
        user_message: User message dict
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        str: Generated text output
    """
    # Process vision info and tokenize
    image_inputs, video_inputs = process_vision_info([system_message, user_message])
    text = processor.apply_chat_template(
        [system_message, user_message], 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        videos=video_inputs, 
        padding=True, 
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Generate prediction
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            temperature=1.0, 
            use_cache=True
        )
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
    
    return output_text


def run_single_inference(model, processor, image, instruction, max_new_tokens=32, 
                        resized_height=None, resized_width=None):
    """
    Run inference on a single image with an instruction.
    
    Args:
        model: The model to use (already loaded)
        processor: AutoProcessor instance
        image: PIL Image
        instruction: Text instruction
        max_new_tokens: Maximum tokens to generate
        resized_height: Optional pre-computed resized height
        resized_width: Optional pre-computed resized width
    
    Returns:
        dict with keys: pred_x_scaled, pred_y_scaled, pred_x, pred_y, output_text, 
                       scale_x, scale_y, original_width, original_height, resized_width, resized_height
    """
    # Prepare image and messages
    resized_image, system_message, user_message, scale_x, scale_y, original_width, original_height = \
        prepare_image_and_messages(image, instruction, processor, resized_height, resized_width)
    
    # Generate prediction
    output_text = generate_prediction(model, processor, system_message, user_message, max_new_tokens)
    
    # Extract and rescale coordinates
    pred_x, pred_y = extract_coordinates(output_text)
    pred_x_scaled = pred_x * scale_x
    pred_y_scaled = pred_y * scale_y
    
    return {
        'pred_x_scaled': pred_x_scaled,
        'pred_y_scaled': pred_y_scaled,
        'pred_x': pred_x,
        'pred_y': pred_y,
        'output_text': output_text,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'original_width': original_width,
        'original_height': original_height,
        'resized_width': resized_width or resized_image.width,
        'resized_height': resized_height or resized_image.height,
    }


def plot_coordinates_on_image(image, pred_x, pred_y, gt_x=None, gt_y=None, 
                              pred_bbox=None, gt_bbox=None,
                              save_path=None, point_radius=5, line_width=2,
                              pred_color='red', gt_color='green', show_text=True):
    """
    Plot predicted (and optionally ground truth) coordinates and bounding boxes on an image.
    
    Args:
        image: PIL Image to annotate
        pred_x: Predicted x coordinate (in image coordinates)
        pred_y: Predicted y coordinate (in image coordinates)
        gt_x: Optional ground truth x coordinate
        gt_y: Optional ground truth y coordinate
        pred_bbox: Optional predicted bounding box [x1, y1, x2, y2]
        gt_bbox: Optional ground truth bounding box [x1, y1, x2, y2]
        save_path: Optional path to save the annotated image
        point_radius: Radius of the point marker
        line_width: Width of crosshair lines and bbox lines
        pred_color: Color for predicted point/bbox (default: 'red')
        gt_color: Color for ground truth point/bbox (default: 'green')
        show_text: Whether to show coordinate text labels
    
    Returns:
        PIL Image: Annotated image
    """
    # Create a copy to avoid modifying the original
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Get image dimensions
    width, height = annotated_image.size
    
    # Ensure coordinates are within image bounds
    pred_x = max(0, min(pred_x, width - 1))
    pred_y = max(0, min(pred_y, height - 1))
    
    # Draw predicted bounding box if provided
    if pred_bbox is not None:
        x1, y1, x2, y2 = pred_bbox
        # Ensure bbox is within image bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=line_width)
        # Add small label
        if show_text:
            try:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", 12)
                    except:
                        font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            bbox_text = f"({int(x1)},{int(y1)})-({int(x2)},{int(y2)})"
            # Position text above the bbox, small and simple
            text_x = x1 + 2
            text_y = max(2, y1 - 14)
            draw.text((text_x, text_y), bbox_text, fill=pred_color, font=font)
    
    # Draw predicted point (small dot)
    point_bbox = [
        pred_x - point_radius,
        pred_y - point_radius,
        pred_x + point_radius,
        pred_y + point_radius
    ]
    draw.ellipse(point_bbox, fill=pred_color, outline=pred_color, width=1)
    
    # Add small text label for predicted coordinates
    if show_text:
        try:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        text = f"({int(pred_x)},{int(pred_y)})"
        # Position text near the point, small and simple
        text_x = pred_x + point_radius + 3
        text_y = pred_y - 6
        draw.text((text_x, text_y), text, fill=pred_color, font=font)
    
    # Draw ground truth point if provided
    if gt_x is not None and gt_y is not None:
        gt_x = max(0, min(gt_x, width - 1))
        gt_y = max(0, min(gt_y, height - 1))
        
        # Draw ground truth bounding box if provided
        if gt_bbox is not None:
            x1, y1, x2, y2 = gt_bbox
            # Ensure bbox is within image bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=line_width)
            # Add small label
            if show_text:
                try:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                    except:
                        try:
                            font = ImageFont.truetype("arial.ttf", 12)
                        except:
                            font = ImageFont.load_default()
                except:
                    font = ImageFont.load_default()
                
                bbox_text = f"({int(x1)},{int(y1)})-({int(x2)},{int(y2)})"
                # Position text below the bbox, small and simple
                text_x = x1 + 2
                text_y = min(height - 16, y2 + 2)
                draw.text((text_x, text_y), bbox_text, fill=gt_color, font=font)
        
        # Draw ground truth point (small dot)
        gt_point_bbox = [
            gt_x - point_radius,
            gt_y - point_radius,
            gt_x + point_radius,
            gt_y + point_radius
        ]
        draw.ellipse(gt_point_bbox, fill=gt_color, outline=gt_color, width=1)
        
        # Add small text label for ground truth
        if show_text:
            try:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", 12)
                    except:
                        font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            gt_text = f"({int(gt_x)},{int(gt_y)})"
            # Position text near the point, small and simple
            gt_text_x = gt_x + point_radius + 3
            gt_text_y = gt_y - 6
            draw.text((gt_text_x, gt_text_y), gt_text, fill=gt_color, font=font)
    
    # Save if path provided
    if save_path:
        annotated_image.save(save_path)
        print(f"Annotated image saved to: {save_path}")
    
    return annotated_image

