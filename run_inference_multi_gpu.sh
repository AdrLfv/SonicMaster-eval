#!/bin/bash

# Multi-GPU inference launcher for SonicMaster
# Usage: bash run_inference_multi_gpu.sh [num_gpus]

NUM_GPUS=${1:-4}

echo "Running inference on $NUM_GPUS GPUs"

torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    inference_ptload_batch.py \
    --model_ckpt checkpoints/model.safetensors
