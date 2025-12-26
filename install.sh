rm -rf $HOME/miniconda3 && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p $HOME/miniconda3 && rm /tmp/miniconda.sh && $HOME/miniconda3/bin/conda init bash
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
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.4/flash_attn-2.8.3%2Bcu128torch2.9-cp310-cp310-linux_aarch64.whl
pip install deepspeed

RUN_NAME=test
torchrun \
    --nproc_per_node 1 \
    --max-restarts 3 \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:29500" src/grpo_grounding.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir grounding/$RUN_NAME \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct"  \
    --dataset_name preprocessing/inp.json \
    --image_root "./preprocessing" \
    --max_prompt_length 1024 \
    --max_completion_length 128 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --freeze_vision_modules true \
    --reward_funcs accuracy \
    --beta 0 \
    --dataloader_num_workers 2 \
    --max_pixels $((4096 * 2160)) \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name output/$RUN_NAME \
    --save_steps 10 \
    --save_total_limit 4 \
    --save_only_model false