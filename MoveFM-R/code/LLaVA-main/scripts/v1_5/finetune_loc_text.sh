#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1500 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
MODEL_VERSION=traj_0820

OUT_DIR=0820_stage0_pretrain_load_7b_100wan
deepspeed  --include localhost:0,1,2,3,4,5,6,7 llava/train/train_mem.py \
    --lora_enable True --lora_r 64 --lora_alpha 128 --lora_dropout 0.0  --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path qwen2_5_7b_with_new_vocab_100wan_code512 \   # The initialization encoding model you trained
    --version qwen2 \
    --data_path data/stage1/s2_final_train_dat_list_id_and_loc_4_city_100wan_new_with_share_gpt.json \
    --image_folder data/stage2/so_final_all_task_data_train.npy  \
    --data_path_eval data/stage1/s2_final_test_dat_list_id_and_loc_4_city_100wan_new_with_share_gpt.json \
    --image_folder_eval data/stage2/so_final_all_task_data_test.npy \
    --vision_tower "traj_model/model_pretrain.pth" \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-$OUT_DIR-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --save_strategy "steps" \
    --save_steps 2000  \
    --save_total_limit 100 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 4800 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --eval_on_start True
