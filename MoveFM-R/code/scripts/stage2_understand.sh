#!/bin/bash

# Train on the trajectory description task
bash code/LLaVA-main/scripts/v1_5/finetune_description.sh

# Train all tasks, including spatiotemporal feature understanding, prediction, and generation
bash code/LLaVA-main/scripts/v1_5/finetune_all_task_under_predict_generate.sh