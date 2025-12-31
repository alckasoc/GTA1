import torch
import re
import os
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.generation import GenerationConfig
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from qwen_vl_utils import process_vision_info

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
        return None, None


class Qwen2_5VLModelNew:
    def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        
        self.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to_dict()
        self.set_generation_config(max_length=2048, do_sample=False, temperature=0.0)

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            assert os.path.exists(image) and os.path.isfile(image), "Invalid input image path."
            image = Image.open(image).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        # Resize image
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=99999999,
        )
        print(f"Resized image size: {resized_width}x{resized_height}")
        resized_image = image.resize((resized_width, resized_height))

        # Build messages with GTA1 prompt format
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT.format(height=resized_height, width=resized_width)}]},
            {"role": "user", "content": [{"type": "image", "image": resized_image}, {"type": "text", "text": instruction}]}
        ]

        # Process and generate
        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        
        print(f"Input length: {len(inputs.input_ids[0])}")
        generated_ids = self.model.generate(**inputs, max_new_tokens=32)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        print(f"Response: {response}")

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # Parse coordinates
        pred_x, pred_y = extract_coordinates(response)
        if pred_x is not None and pred_y is not None:
            print(f"Parsed coordinates: ({pred_x}, {pred_y})")
            result_dict["point"] = [pred_x / resized_width, pred_y / resized_height]

        return result_dict

    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError()

