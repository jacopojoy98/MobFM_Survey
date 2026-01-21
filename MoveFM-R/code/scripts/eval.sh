#! /bin/bash



# Pay attention to replacing the corresponding model path and test dataset path

# Generate responses for the test set prediction task (Note to replace the dataset path)
bash code/LLaVA-main/scripts/v1_5/finetune_eval_qwen.sh

# Evaluation Accuracy
python  code/LLaVA-main/eval_traj/cal_acc_predict.py

# Generate the response for the test set generation task unconditionally (note to replace the dataset path)
bash code/LLaVA-main/scripts/v1_5/finetune_eval_qwen.sh

# Add self-reflection reinforcement, conditional generation
CUDA_VISIBLE_DEVICES=0  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1500  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   llamafactory-cli train code/LLaMA-Factory-main/examples/inference/qwen_traj_rl_gene.yaml

# Evaluate JSD,TVD,BLEU
python code/LLaVA-main/eval_traj/eval_gene.py