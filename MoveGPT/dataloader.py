from torch.utils.data.distributed import DistributedSampler
import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random

class TrajDataset(Dataset):
    def __init__(self, data_root, split, B,T,few_shot):
        self.B = B
        self.T = T
        self.split = split
        self.few_shot = few_shot
        
        # load the shards
        shards = os.listdir(data_root)
        shards = [s for s in shards if any(x in s for x in split)]
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        
        self.data_city = defaultdict()
        for shard in self.shards:
            self.data_city[shard] = self.load_traj(shard)
        self.data = []

        # 获取每个key的数据批次
        batches = {shard: [self.data_city[shard][i:i + self.B] 
                           for i in range(0, len(self.data_city[shard]), self.B)] 
                   for shard in self.shards}



        # 获取所有 shard 中的所有 batch 的总数
        total_batches = sum(len(batches[shard]) for shard in self.shards)

        # 创建一个记录每个 shard 中已取 batch 的索引的字典
        shard_indices = {shard: 0 for shard in self.shards}

        # 创建一个记录每个 shard 中剩余 batch 数量的字典
        remaining_batches = {shard: len(batches[shard]) for shard in self.shards}

        # 随机取 batch，直到所有 batch 都被取完
        for _ in range(total_batches):
            # 随机选择一个 shard
            shard = random.choice(self.shards)
            
            # 如果该 shard 中还有剩余的 batch
            if remaining_batches[shard] > 0:
                # 获取当前 shard 中的下一个 batch
                batch = batches[shard][shard_indices[shard]]
                # 将 batch 数据添加到 self.data 中
                self.data.extend(batch)
                # 更新 shard 的索引和剩余 batch 数量
                shard_indices[shard] += 1
                remaining_batches[shard] -= 1
                
    def load_traj(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            size = int(len(lines)/self.B)*self.B
            lines = lines[:size]
            data = []
            for line in lines:
                traj = []
                line = line.strip()
                userid = line.split(' ')[0]
                trajs = line.split(' ')[1]
                parts = trajs.strip().split(';')
                for part in parts:
                    if part:  # 确保不处理空字符串
                        location, day, time = part.split(',')
                        day = int(day)
                        if day == 0:
                            day = 7
                        time = int(time)
                        traj.append([int(location) + 2, time,day])
                # 确保traj的长度至少为144，不足的地方用[0,0]补全
                traj.append([int(1), int(0),int(0)])
                for _ in range(self.T+1 - len(traj)):
                    traj.append([int(0), int(0), int(0)])
                traj = torch.tensor(traj, dtype=torch.long)
                data.append([traj,filename.split('/')[-1].split('_')[0]])
            if self.few_shot:
                length = int(self.few_shot*len(data))
                data = data[:length]
            return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj,file = self.data[idx]             
        x = traj[:-1, 0]
        y = traj[1:, 0]
        ts_his = traj[:-1, 1]
        ts_next = traj[1:,1]
        day_his = traj[:-1,2]
        day_next = traj[1:,2]
        condition = day_his == day_next
        condition = ~condition
        zero_matrix = day_his * 0
        result_weekday = torch.where(condition, 48, zero_matrix)
        stay_time = result_weekday + ts_next - ts_his
        return x, y, ts_his,day_his,stay_time,file



def get_dataloader_ddp(data_root, split, B, T, few_shot):
    dataset = TrajDataset(data_root, split, B, T, few_shot)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=False, sampler=sampler)
    return dataloader, sampler



class GroupedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, B=2):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.B = B  # 每组的大小

    def __iter__(self):
        # 获取整个数据集的索引
        indices = list(range(len(self.dataset)))

        # 按相邻的 B 个元素分组
        groups = [indices[i:i + self.B] for i in range(0, len(indices), self.B)]

        # 选择当前进程对应的组
        selected_groups = groups[self.rank::self.num_replicas]

        # 将选择的组展平为一个列表
        selected_indices = [item for group in selected_groups for item in group]

        # 确保每个进程的样本数量一致（避免最后一个组不足的情况）
        # 如果需要，可以填充一些索引
        max_length = max(len(selected_indices), self.num_samples)
        if len(selected_indices) < max_length:
            padding = [selected_indices[i % len(selected_indices)] for i in range(max_length - len(selected_indices))]
            selected_indices += padding

        return iter(selected_indices)

    def __len__(self):
        # 计算每个进程的样本数量
        return len(self.dataset) // self.num_replicas