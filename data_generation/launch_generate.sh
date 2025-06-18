#!/bin/bash

python generate.py \
--data_path /home/nxclab/wonjun/Medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
--output_path /home/nxclab/wonjun/Medusa/ShareGPT_Vicuna_unfiltered/train_shareGPT_llama3.2_1B.jsonl \
--num_threads 256 \
--temperature 0.3 \
--max_tokens 512
