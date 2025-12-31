# ARM64
# rm -rf $HOME/miniconda3 && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p $HOME/miniconda3 && rm /tmp/miniconda.sh && $HOME/miniconda3/bin/conda init bash
# X86_64
rm -rf $HOME/miniconda3 && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p $HOME/miniconda3 && rm /tmp/miniconda.sh && $HOME/miniconda3/bin/conda init bash
source ~/.bashrc

conda create -n vlm-r1 python=3.10
conda activate vlm-r1

pip install \
    ninja \
    packaging \
    psutils \
    peft \
    datasets \
    transformers \
    trl \
    liger-kernel \
    wandb \
    tensorboardx \
    qwen_vl_utils \
    babel \
    python-Levenshtein \
    matplotlib \
    pycocotools \
    openai \
    math-verify \
    "httpx[socks]"
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
# https://flashattn.dev/
# ARM64
# pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.4/flash_attn-2.8.3%2Bcu128torch2.9-cp310-cp310-linux_aarch64.whl
# X86_64
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp310-cp310-linux_x86_64.whl
pip install deepspeed

wandb login

sudo rm -rf grounding

# download dataset
mkdir data
mkdir data/images
# check tmp.ipynb to download dataset
cd data
cat image.part.* > image.zip
mv image.zip images/image.zip
cd images
unzip image.zip
rm image.zip
cd ../..

RUN_NAME=distance_reward
WANDB_PROJECT=gta1
torchrun \
    --nproc_per_node 2 \
    --max-restarts 3 \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:29500" src/grpo_grounding.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir grounding/$RUN_NAME \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct"  \
    --dataset_name images/inp_copy.json \
    --image_root "./images/" \
    --max_prompt_length 1024 \
    --max_completion_length 128 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --freeze_vision_modules true \
    --reward_funcs distance \
    --beta 0 \
    --dataloader_num_workers 2 \
    --max_pixels $((4096 * 2160)) \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 100 \
    --run_name $RUN_NAME \
    --save_steps 10 \
    --save_total_limit 4

# Configure args inside script.
python src/run_eval.py

# Upload model to Hugging Face.
huggingface-cli login
huggingface-cli upload alckasoc/mini-gta1-3b /home/ubuntu/GTA1/grounding/distance_reward/checkpoint-80 checkpoint-80-best-distance-reward