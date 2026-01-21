import os
import numpy as np
import torch
from torch.nn import functional as F
torch.cuda.empty_cache()  # Clear the cache
from dataloader import TrajDataset, DistributedSampler,GroupedDistributedSampler
import random
import argparse
from torch.utils.data import DataLoader
from utils import train_stop_ddp, evaluate_ddp,train_tune
import torch.distributed as dist
from model import Traj_Config, Traj_Model
from torch.nn.parallel import DistributedDataParallel as DDP


def set_random_seed(seed: int):
    """
    固定随机种子以确保结果的可重复性

    参数:
    seed (int): 要设置的随机种子
    """
    random.seed(seed)                        # Python 随机数生成器
    np.random.seed(seed)                     # NumPy 随机数生成器
    torch.manual_seed(seed)                  # PyTorch 随机数生成器（CPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)         # PyTorch 随机数生成器（当前 GPU）
        torch.cuda.manual_seed_all(seed)     # PyTorch 随机数生成器（所有 GPU）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main_ddp():
    # 解析 --local-rank 参数
    parser = argparse.ArgumentParser(description="traj_MOE_MODEL")
    # 动态检测当前进程的 local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        raise ValueError("LOCAL_RANK environment variable not set. Please use torchrun or similar tools to launch the script.")

    # 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 设置随机种子
    set_random_seed(520)

    # 其他参数解析
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--B", type=int, default=8, help='batch size')
    parser.add_argument("--T", type=int, default=144, help='三天轨迹最大长度')
    parser.add_argument("--fcity", nargs='+', default=['Atlanta'])   
    parser.add_argument("--city", nargs='+', default=['Atlanta'])
    parser.add_argument("--target_city", nargs='+', default=['Atlanta'])
    parser.add_argument("--few_trans", type=str, default='1.0')
    parser.add_argument("--train_root", type=str, default='./traj_dataset/train')
    parser.add_argument("--val_root", type=str, default='./traj_dataset/val')
    parser.add_argument("--test_root", type=str, default='./traj_dataset/test')
    parser.add_argument("--few_shot", type=float, default=1.0)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4, help='learning rate')
    parser.add_argument("--model", type=str, default='moe_dcn', help='model')
    args = parser.parse_args()

    log_dir = f"{args.city}/{args.n_layer}_{args.n_embd}"
    os.makedirs(log_dir, exist_ok=True)

    # 创建模型并包装为 DDP
    model = Traj_Model(Traj_Config(n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer)).to(device)
    
    model = DDP(
    model,
    device_ids=[device],
    output_device=device,
    find_unused_parameters=True)  # 启用未使用参数检测
 
    model.module.load_state_dict(torch.load(f"./model_pretrain.pth", map_location=device))
    # print(type(model))  # 应该输出 <class 'torch.nn.parallel.distributed.DistributedDataParallel'>
    # print(type(model.module))  # 应该输出你的模型类，如 <class '__main__.Traj_Model'>
    # 创建数据加载器
    log_dir = log_dir + f'/{args.few_trans}'
    os.makedirs(log_dir, exist_ok=True)
    # train_root  = f"{args.train_root}/{args.few_trans}/{args.city[0]}/train"
    train_root = args.train_root
    train_dataset = TrajDataset(train_root, args.city, args.B, args.T, args.few_shot)
    train_sampler = GroupedDistributedSampler(train_dataset,shuffle=False,B=args.B)
    train_loader = DataLoader(train_dataset, batch_size=args.B, sampler=train_sampler,shuffle=False)

    val_loaders = []
    for city in args.city:
        city = [f'{city}']
        val_dataset = TrajDataset(args.val_root, city, args.B, args.T, args.few_shot)
        val_sampler = DistributedSampler(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=args.B, sampler=val_sampler)
        val_loaders.append(val_loader)
    valid_step_interval = len(train_dataset) // args.B //args.rank # 每训练1/4个epoch验证一次模型

    # 训练和验证
    train_tune(model, train_loader, val_loaders, log_dir, args.lr, args.epoch, valid_step_interval, device, args.city, patience=3)

    # 清理分布式环境
    dist.destroy_process_group()


if __name__ == "__main__":
    main_ddp()
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node=4 train_ddp.py