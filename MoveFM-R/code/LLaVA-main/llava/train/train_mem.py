from llava.train.train import train
import random
import  numpy as np
import torch
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
if __name__ == "__main__":
    seed =42
    set_all_seeds(seed)
    print("setting seed=",seed)
    train(attn_implementation="flash_attention_2")
