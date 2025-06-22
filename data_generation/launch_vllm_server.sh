#!/bin/bash

MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME --port 8000