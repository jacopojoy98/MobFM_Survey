#! /bin/bash

# Train the RQ-VAE codebook
python code/gene_index/main.py


# Generate RQ-VAE codebooks using the trained model
python code/gene_index/generate_indices.py


# Optimize the initial embedding vector of the token corresponding to the codebook
python code/gene_index/token_initialization.py


#Train the model on a question-answer matching task involving millions of location-ids and geographic information
bash code/LLaVA-main/scripts/v1_5/finetune_lora.sh