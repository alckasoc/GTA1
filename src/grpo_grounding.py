# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math
from qwen_vl_utils import smart_resize, process_vision_info
from transformers import AutoProcessor
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
import torch
import wandb
from utils.inference_utils import run_single_inference, extract_coordinates, plot_coordinates_on_image

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

SYSTEM_PROMPT = '''
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
'''
SYSTEM_PROMPT = SYSTEM_PROMPT.strip()

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments, processing_class:None):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        cur_data_dict = []
        if data_path.endswith(".json"):
            with open(data_path, "r") as json_file:
                self.list_data_dict = json.load(json_file)
        else:
            self.list_data_dict = []
            with open(data_path, "r") as f:
                for line in f:
                    self.list_data_dict.append(json.loads(line))
        self.processor=processing_class
        
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        def make_conversation_image(example, height,width):
            instruction = example['conversations'][0]['value']
            instruction = instruction.replace("<image>","")
            return {
                "prompt": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT.format(height=height,width=width)}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": instruction}
                        ],
                    },
                ],
            }
        example = self.list_data_dict[i]
        image_root = self.script_args.image_root   
        image_path = os.path.join(image_root, example['image'])
        image = Image.open(image_path).convert("RGB")
        image_height, image_width =  image.height, image.width
        resized_height, resized_width  = smart_resize(
                    image.height,
                    image.width,
                    factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
                    min_pixels=self.processor.image_processor.min_pixels,
                    max_pixels=self.processor.image_processor.max_pixels,
                    )
        image = image.resize((resized_width, resized_height))
        box = example['bbox']
        image = [image]
        

        solution = [box,resized_height/1000,resized_width/1000]

        return {
            'image': image,
            'problem': example['conversations'][0]['value'],
            'solution': solution,
            'prompt': make_conversation_image(example, resized_height,resized_width)['prompt']
        }

def parse_coordinates(raw_string):
    matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", raw_string)
    matches = [tuple(map(int, match)) for match in matches]
    if len(matches) > 1:
        return -1,-1
    else:
        return matches[0]


def click_reward(completions, solution, **kwargs):
    def isin(x,y, sol):
        boxs,ratio_h,ratio_w=sol[:3]
        if not isinstance(boxs[0], list):
            boxs = [boxs]
        for box in boxs:
            x0,y0,x1,y1 = box
            x0 = int(x0*ratio_w)
            y0 = int(y0*ratio_h)
            x1 = int(x1*ratio_w)
            y1 = int(y1*ratio_h)
            if x<=x1 and x>=x0:
                if y<=y1 and y>=y0:
                    return True
        return False
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            pred_x, pred_y = parse_coordinates(content)
            if isin(pred_x,pred_y, sol):
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>\(\d+,\s*\d+\)</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content.strip(), re.DOTALL) for content in completion_contents]
    reward=[1.0 if match else 0.0 for match in matches]
    return reward

reward_funcs_registry = {
    "accuracy": click_reward,
    "format": format_reward,
}


def evaluate_and_log_to_wandb(trainer, dataset, processing_class, script_args, num_examples=1):
    """Run inference on a subset of examples and log results to wandb table."""
    run = wandb.init(project="huggingface", name="eval")
    
    model = trainer.model
    model.eval()
    
    # Get subset of examples (don't request more than available)
    num_available = len(dataset)
    num_to_eval = min(num_examples, num_available)
    if num_to_eval == 0:
        print(f"Warning: Dataset has 0 examples, skipping evaluation")
        return
    
    eval_indices = list(range(num_available))
    # random.seed(42)
    random.shuffle(eval_indices)
    eval_indices = eval_indices[:num_to_eval]
    print(f"Evaluating on {num_to_eval} example(s) (dataset has {num_available} total)")
    
    table_data = []

    print(f"Evaluating on {eval_indices} examples")
    
    with torch.no_grad():
        for idx in eval_indices:
            example = dataset[idx]
            image = example['image'][0]  # Get the PIL Image
            prompt = example['prompt']
            solution = example['solution']
            problem = example['problem']
            
            # Get original image dimensions for scaling
            original_image_path = os.path.join(script_args.image_root, dataset.list_data_dict[idx]['image'])
            original_image = Image.open(original_image_path).convert("RGB")
            original_width, original_height = original_image.width, original_image.height
            
            # Get resized dimensions
            resized_height, resized_width = image.height, image.width
            
            # Get instruction from problem
            instruction = problem.replace("<image>", "").strip()
            
            # Run inference using shared utility
            results = run_single_inference(
                model=model,
                processor=processing_class,
                image=original_image,  # Use original image, utility will resize
                instruction=instruction,
                max_new_tokens=32,
                resized_height=resized_height,
                resized_width=resized_width
            )
            
            output_text = results['output_text']
            pred_x = results['pred_x']
            pred_y = results['pred_y']
            pred_x_scaled = results['pred_x_scaled']
            pred_y_scaled = results['pred_y_scaled']
            scale_x = results['scale_x']
            scale_y = results['scale_y']
            
            # Get ground truth box
            # The bbox in solution is stored as absolute pixel coordinates from the JSON
            # Get it directly from the dataset's list_data_dict for accuracy
            bbox_from_json = dataset.list_data_dict[idx]['bbox']
            x0, y0, x1, y1 = bbox_from_json
            
            # Bbox is already in original image coordinates
            gt_bbox_original = [x0, y0, x1, y1]
            # Get center point in original coordinates
            gt_center_x = (x0 + x1) / 2
            gt_center_y = (y0 + y1) / 2
            
            # Also get from solution for correctness check (in resized coordinates)
            boxs, ratio_h, ratio_w = solution[:3]
            if not isinstance(boxs[0], list):
                boxs = [boxs]
            
            # Check if prediction is correct (in resized coordinates)
            is_correct = False
            for box in boxs:
                x0, y0, x1, y1 = box
                x0_scaled = int(x0 * ratio_w * resized_width)
                y0_scaled = int(y0 * ratio_h * resized_height)
                x1_scaled = int(x1 * ratio_w * resized_width)
                y1_scaled = int(y1 * ratio_h * resized_height)
                if pred_x <= x1_scaled and pred_x >= x0_scaled:
                    if pred_y <= y1_scaled and pred_y >= y0_scaled:
                        is_correct = True
                        break
            
            # Create annotated image with predictions and ground truth
            annotated_image = plot_coordinates_on_image(
                image=original_image,
                pred_x=pred_x_scaled,
                pred_y=pred_y_scaled,
                gt_x=gt_center_x,
                gt_y=gt_center_y,
                gt_bbox=gt_bbox_original,
                show_text=True
            )
            
            # Format bbox as string
            gt_bbox_str = f"[{int(x0)}, {int(y0)}, {int(x1)}, {int(y1)}]"
            
            # Add to table data
            table_data.append([
                idx,
                wandb.Image(annotated_image, caption=f"Example {idx}"),
                f"({pred_x_scaled:.1f}, {pred_y_scaled:.1f})",
                f"({gt_center_x:.1f}, {gt_center_y:.1f})",
                gt_bbox_str
            ])


    print(f"Table data: {table_data}")
    if table_data:
        # Create wandb table with updated columns
        table_columns = ["Index", "Annotated Image", "Predicted Coordinate", "Ground Truth Coordinate", "Ground Truth Bbox"]
        print(f"Creating wandb table with columns: {table_columns}")
        print(f"Number of data rows: {len(table_data)}, Number of columns: {len(table_columns)}")
        table = wandb.Table(
            columns=table_columns,
            data=table_data
        )
        
        # Log to wandb (will use the same run that trainer initialized)
        run.log({"somethign else": table})
        print(f"Logged evaluation results for {len(table_data)} example(s) to wandb run: {wandb.run.name if wandb.run else 'unknown'}")
    else:
        print(f"Warning: No evaluation data collected. Tried {len(eval_indices)} example(s) but all failed.")
    
    run.finish()
    model.train()


def main(script_args, training_args, model_args):

    print(script_args)
    print(training_args)
    print(model_args)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    processing_class = AutoProcessor.from_pretrained(model_args.model_name_or_path,  max_pixels=script_args.max_pixels, min_pixels=script_args.min_pixels)
    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args, processing_class)
    trainer_cls = Qwen2VLGRPOTrainer

    apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )


    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    
    # Run evaluation and log to wandb
    if training_args.report_to and "wandb" in training_args.report_to:
        print("Running evaluation on subset of examples...")
        evaluate_and_log_to_wandb(trainer, dataset, processing_class, script_args, num_examples=1)
    
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
