CUDA_VISIBLE_DEVICES=0,1 python eval_screenspot_pro_parallel.py  \
    --max_samples 100 \
    --model_type "qwen2_5vl_new" \
    --model_name_or_path "/home/ubuntu/GTA1/grounding/test/checkpoint-80-best" \
    --screenspot_imgs "/home/ubuntu/ScreenSpot-Pro-GUI-Grounding/data/images" \
    --screenspot_test "/home/ubuntu/ScreenSpot-Pro-GUI-Grounding/data/annotations" \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --log_path "./results/gta1_baseline_80.json" \
    --inst_style "instruction"

CUDA_VISIBLE_DEVICES=0,1 python eval_screenspot_pro_parallel.py  \
    --max_samples 100 \
    --model_type "qwen2_5vl_new" \
    --model_name_or_path "/home/ubuntu/GTA1/grounding/distance_reward/checkpoint-80" \
    --screenspot_imgs "/home/ubuntu/ScreenSpot-Pro-GUI-Grounding/data/images" \
    --screenspot_test "/home/ubuntu/ScreenSpot-Pro-GUI-Grounding/data/annotations" \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --log_path "./results/gta1_distance_reward_80.json" \
    --inst_style "instruction"

CUDA_VISIBLE_DEVICES=0,1 python eval_screenspot_pro_parallel.py  \
    --max_samples 100 \
    --model_type "qwen2_5vl" \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --screenspot_imgs "/home/ubuntu/ScreenSpot-Pro-GUI-Grounding/data/images" \
    --screenspot_test "/home/ubuntu/ScreenSpot-Pro-GUI-Grounding/data/annotations" \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --log_path "./results/base_model_3b.json" \
    --inst_style "instruction"