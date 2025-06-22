#!/bin/bash

docker run --gpus all -it --rm \
           --shm-size=16g \
           -v $HOME/.cache/huggingface:/root/.cache/huggingface \
           june0912/medusa:0.2.0       # image tag you built
# you’ll land in /bin/bash (your Dockerfile’s CMD)
