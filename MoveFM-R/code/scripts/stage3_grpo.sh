#!/bin/bash

# SFT pre-training
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1500 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True llamafactory-cli train code/LLaMA-Factory-main/examples/train_lora/qwen3_lora_qwen_traj_rl_sft.yaml

# GRPO self-reflective reasoning training
bash code/verl/examples/grpo_trainer/run_movefm_rl.sh