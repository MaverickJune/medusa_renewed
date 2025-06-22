#! /bin/bash

MODEL_NAME=meta-llama/Llama-3.2-1B
DATA_PATH=/home/nxclab/wonjun/Medusa/ShareGPT_Vicuna_unfiltered/train_shareGPT_llama3.2_1B.jsonl

torchrun --nproc_per_node=2 medusa/train/train_legacy.py --model_name_or_path $MODEL_NAME \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir test \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 3 \
    --save_safetensors False \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --lazy_preprocess True \
    --medusa_num_heads 2 \
    --medusa_num_layers 1 \
    --deepspeed deepspeed.json