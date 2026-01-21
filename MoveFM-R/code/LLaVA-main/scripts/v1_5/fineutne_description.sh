#!/bin/bash
MODEL_VERSION=0820_stage1_pretrain_load_7b_gai_right_code512_new_bf16
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1500 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

deepspeed --include localhost:0,1,2,3,4,5,6,7 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path qwen2_5_7b_with_new_vocab_100wan_code512 \   # The initialization encoding model you trained
    --version plain \
    --data_path data/stage2/s0_duiqi_final_train_textandemb_loc2id_4_city_all_tasks.json \
    --image_folder data/stage2/trajectory_description_train.npy  \
    --data_path_eval data/stage2/s0_duiqi_final_test_textandemb_loc2id_4_city_all_tasks.json \
    --image_folder_eval data/stage2/trajectory_description_test.npy  \
    --vision_tower "traj_model/model_pretrain.pth" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-pretrain \
    --num_train_epochs 3 \
    --per_device_train_batch_size  12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 1500 \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_total_limit 1000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 4500 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb  \
    --eval_on_start False

