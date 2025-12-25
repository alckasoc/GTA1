wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p $HOME/miniconda3 && rm /tmp/miniconda.sh && $HOME/miniconda3/bin/conda init bash

conda create -n vlm-r1 python=3.10
conda activate vlm-r1

pip install psutils
pip install peft
pip install datasets
pip install transformers trl
pip install liger-kernel
pip install wandb
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation
pip install babel
pip install python-Levenshtein
pip install matplotlib
pip install pycocotools
pip install openai
pip install httpx[socks]

python -c "from datasets import load_dataset; ds = load_dataset('HelloKKMe/grounding_dataset')"

RUN_NAME=test
srun torchrun \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node 8 \
    --max-restarts 3 \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint "$RDZV_HOST:$RDZV_PORT"  src/grpo_grounding.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir grounding/$RUN_NAME \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct"  \
    --dataset_name preprocessing/inp.json \
    --image_root "./preprocessing" \
    --max_prompt_length 1024 \
    --max_completion_length 128 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --freeze_vision_modules true \
    --reward_funcs accuracy \
    --beta 0 \
    --dataloader_num_workers 2 \
    --max_pixels $((4096 * 2160)) \
    --gradient_accumulation_steps 32 \
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