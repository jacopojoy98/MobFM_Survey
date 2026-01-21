import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from tqdm import tqdm
import pickle
import os
import json
import torch.nn.functional as F


def train_stop_ddp(model, train_loader, valid_loaders, log_dir, lr, epoch, valid_step_interval, device, citys, patience=10):
    # 初始化分布式环境
    #dist.init_process_group(backend='nccl')
    # model = DDP(model, device_ids=[device], output_device=device)
    
    # 创建日志文件
    if dist.get_rank() == 0:
        log_file_train = os.path.join(log_dir, f"log_train.txt")
        with open(log_file_train, "w") as f:
            pass

    # 定义优化器和学习率调度器
    optimizer = model.module.configure_optimizers(weight_decay=0.1, learning_rate=lr)
    
    p1 = int(0.4 * epoch)
    p2 = int(0.6 * epoch)
    p3 = int(0.9 * epoch)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2, p3], gamma=0.4
    )
    
    best_valid_loss_dict = {}
    patience_counter_dict = {}
    for city in citys:
        best_valid_loss_dict[f'{city}'] = 1e10
        patience_counter_dict[f'{city}'] = 0
        if dist.get_rank() == 0:
            log_file_val = os.path.join(log_dir, f"log_val_{city}.txt")
            with open(log_file_val, "w") as f:
                pass

    for epoch_no in range(epoch):
        train_loader.sampler.set_epoch(epoch_no)
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0, disable=dist.get_rank() != 0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                x_train = train_batch[0]
                y_train = train_batch[1]
                ts = train_batch[2]
                dow = train_batch[3]
                stay_time = train_batch[4]
                train_city = train_batch[5][0]
                vocab = np.load(f'./location_feature/vocab_{train_city}.npy')
                vocab = np.pad(vocab, ((2,0), (0, 0)), mode='constant', constant_values=0)
                vocab = torch.from_numpy(vocab)
                vocab = vocab.to(torch.float32)
                poi = np.take(vocab, x_train, axis=0) 
                output = model(x_train,poi, ts, dow, vocab, device, y_train, stay_time)
                loss = output['loss'] 
                loss.backward()
                avg_loss += loss.item()
                loss_avg = avg_loss / batch_no
                optimizer.step()
                if dist.get_rank() == 0:
                    with open(log_file_train, "a") as f:
                        f.write(f"{epoch_no}\t{batch_no}\t train \t{loss_avg:.6f}\n")
                    it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss": loss_avg,
                            "epoch": epoch_no,
                        },
                        refresh=False,
                    )

                if valid_loaders is not None and (batch_no + 1) % valid_step_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        for valid_loader in valid_loaders:
                            avg_loss_valid = 0
                            with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0, disable=dist.get_rank() != 0) as it:
                                for batch_no_val, valid_batch in enumerate(it, start=1):
                                    x_val = valid_batch[0]
                                    y_val = valid_batch[1]
                                    ts = valid_batch[2]
                                    dow = valid_batch[3]
                                    stay_time = valid_batch[4]
                                    val_city = valid_batch[5][0]
                                    vocab = np.load(f'./location_feature/vocab_{val_city}.npy')
                                    vocab = np.pad(vocab, ((2, 0), (0, 0)), mode='constant', constant_values=0)
                                    vocab = torch.from_numpy(vocab)
                                    vocab = vocab.to(torch.float32)
                                    poi = np.take(vocab,x_val,axis=0)
                                    output = model(x_val, poi,ts, dow, vocab, device, y_val, stay_time)
                                    loss = output['loss']
                                    avg_loss_valid += loss.item()
                                    loss_avg_valid = avg_loss_valid / batch_no_val
                                    if dist.get_rank() == 0:
                                        it.set_postfix(
                                            ordered_dict={
                                                "valid_avg_loss": loss_avg_valid,
                                                "epoch": epoch_no,
                                            },
                                            refresh=False,
                                        )
                                # 聚合验证结果
                                loss_avg_valid = torch.tensor(loss_avg_valid, device=device)
                                dist.all_reduce(loss_avg_valid, op=dist.ReduceOp.SUM)
                                loss_avg_valid /= dist.get_world_size()
                                if dist.get_rank() == 0:
                                    log_file_val = os.path.join(log_dir, f"log_val_{val_city}.txt")
                                    with open(log_file_val, "a") as f:
                                        f.write(f"{epoch_no}\t{batch_no}\t val \t{loss_avg_valid.item():.6f}\n")

                                    if best_valid_loss_dict[f'{val_city}'] > loss_avg_valid.item():
                                        output_path = log_dir + f"/model_pretrain.pth"
                                        torch.save(model.module.state_dict(), output_path)
                                        best_valid_loss_dict[f'{val_city}'] = loss_avg_valid.item()
                                        patience_counter_dict[f'{val_city}'] = 0  # Reset patience counter
                                        print(
                                            "\n best loss is updated to ",
                                            loss_avg_valid.item() ,
                                            "at",
                                            epoch_no, val_city
                                        )
                                    else:
                                        patience_counter_dict[f'{val_city}'] += 1
                                        if all(value >= patience for value in patience_counter_dict.values()):
                                            print(f"\n Early stopping triggered for {val_city} at epoch {epoch_no}")

                                            return  # Stop training 

        lr_scheduler.step()

def train_tune(model, train_loader, valid_loaders, log_dir, lr, epoch, valid_step_interval, device, citys, patience=10):
    # 初始化分布式环境
    #dist.init_process_group(backend='nccl')
    # model = DDP(model, device_ids=[device], output_device=device)
    
    # 创建日志文件
    if dist.get_rank() == 0:
        log_file_train = os.path.join(log_dir, f"log_train.txt")
        with open(log_file_train, "w") as f:
            pass

    # 定义优化器和学习率调度器
    optimizer = model.module.configure_optimizers(weight_decay=0.1, learning_rate=lr)
    
    p1 = int(0.4 * epoch)
    p2 = int(0.6 * epoch)
    p3 = int(0.9 * epoch)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2, p3], gamma=0.4
    )
    
    best_valid_loss_dict = {}
    patience_counter_dict = {}
    for city in citys:
        best_valid_loss_dict[f'{city}'] = 1e10
        patience_counter_dict[f'{city}'] = 0
        if dist.get_rank() == 0:
            log_file_val = os.path.join(log_dir, f"log_val_{city}.txt")
            with open(log_file_val, "w") as f:
                pass

    for epoch_no in range(epoch):
        train_loader.sampler.set_epoch(epoch_no)
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0, disable=dist.get_rank() != 0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                x_train = train_batch[0]
                y_train = train_batch[1]
                ts = train_batch[2]
                dow = train_batch[3]
                stay_time = train_batch[4]
                train_city = train_batch[5][0]
                vocab = np.load(f'./location_feature/vocab_{train_city}.npy')
                vocab = np.pad(vocab, ((2,0), (0, 0)), mode='constant', constant_values=0)
                vocab = torch.from_numpy(vocab)
                vocab = vocab.to(torch.float32)
                poi = np.take(vocab, x_train, axis=0) 
                output = model(x_train,poi, ts, dow, vocab, device, y_train, stay_time)
                loss = output['loss'] 
                loss.backward()
                avg_loss += loss.item()
                loss_avg = avg_loss / batch_no
                optimizer.step()
                if dist.get_rank() == 0:
                    with open(log_file_train, "a") as f:
                        f.write(f"{epoch_no}\t{batch_no}\t train \t{loss_avg:.6f}\n")
                    it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss": loss_avg,
                            "epoch": epoch_no,
                        },
                        refresh=False,
                    )

                if valid_loaders is not None and (batch_no + 1) % valid_step_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        for valid_loader in valid_loaders:
                            avg_loss_valid = 0
                            with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0, disable=dist.get_rank() != 0) as it:
                                for batch_no_val, valid_batch in enumerate(it, start=1):
                                    x_val = valid_batch[0]
                                    y_val = valid_batch[1]
                                    ts = valid_batch[2]
                                    dow = valid_batch[3]
                                    stay_time = valid_batch[4]
                                    val_city = valid_batch[5][0]
                                    vocab = np.load(f'./location_feature/vocab_{val_city}.npy')
                                    vocab = np.pad(vocab, ((2, 0), (0, 0)), mode='constant', constant_values=0)
                                    vocab = torch.from_numpy(vocab)
                                    vocab = vocab.to(torch.float32)
                                    poi = np.take(vocab,x_val,axis=0)
                                    output = model(x_val, poi,ts, dow, vocab, device, y_val, stay_time)
                                    loss = output['loss']
                                    avg_loss_valid += loss.item()
                                    loss_avg_valid = avg_loss_valid / batch_no_val
                                    if dist.get_rank() == 0:
                                        it.set_postfix(
                                            ordered_dict={
                                                "valid_avg_loss": loss_avg_valid,
                                                "epoch": epoch_no,
                                            },
                                            refresh=False,
                                        )
                                # 聚合验证结果
                                loss_avg_valid = torch.tensor(loss_avg_valid, device=device)
                                dist.all_reduce(loss_avg_valid, op=dist.ReduceOp.SUM)
                                loss_avg_valid /= dist.get_world_size()
                                if dist.get_rank() == 0:
                                    log_file_val = os.path.join(log_dir, f"log_val_{val_city}.txt")
                                    with open(log_file_val, "a") as f:
                                        f.write(f"{epoch_no}\t{batch_no}\t val \t{loss_avg_valid.item():.6f}\n")

                                    if best_valid_loss_dict[f'{val_city}'] > loss_avg_valid.item():
                                        output_path = log_dir + f"/model_{val_city}.pth"
                                        torch.save(model.module.state_dict(), output_path)
                                        best_valid_loss_dict[f'{val_city}'] = loss_avg_valid.item()
                                        patience_counter_dict[f'{val_city}'] = 0  # Reset patience counter
                                        print(
                                            "\n best loss is updated to ",
                                            loss_avg_valid.item() ,
                                            "at",
                                            epoch_no, val_city
                                        )
                                    else:
                                        patience_counter_dict[f'{val_city}'] += 1
                                        if all(value >= patience for value in patience_counter_dict.values()) and epoch_no>19:
                                            print(f"\n Early stopping triggered for {val_city} at epoch {epoch_no}")

                                            return  # Stop training 

        lr_scheduler.step()


def evaluate_ddp(model, test_loader, log_dir, B, city, device):

    # 创建日志文件
    if dist.get_rank() == 0:

        log_file_test = os.path.join(log_dir, f"log_{city[0]}_test.txt")
        with open(log_file_test, "w") as f:
            pass
    
    model.module.load_state_dict(torch.load(log_dir + f"/model_{city[0]}.pth", map_location=device))
    model.eval()
    acc_time = 0
    mse_time = 0
    acc1 = 0
    acc3 = 0
    acc5 = 0
    acc10 = 0
    size = 0
    mean = 0
    val_loss_accum = 0.0
    var = 0
    batch = 0
    with tqdm(test_loader, mininterval=5.0, maxinterval=50.0, disable=dist.get_rank() != 0) as it:
        for batch_no, test_batch in enumerate(it, start=1):
            batch = batch_no + 1
            x_test = test_batch[0]
            y_test = test_batch[1]
            ts = test_batch[2]
            dow = test_batch[3]
            stay_time = test_batch[4]
            test_city = test_batch[5][0]
            vocab = np.load(f'./location_feature/vocab_{test_city}.npy')
            vocab = np.pad(vocab, ((2,0), (0, 0)), mode='constant', constant_values=0)
            vocab = torch.from_numpy(vocab)
            vocab = vocab.to(torch.float32)
            poi = np.take(vocab,x_test,axis=0)
            output = model(x_test,poi, ts, dow, vocab, device, y_test, stay_time)
            loss = output['loss'] 
            val_loss_accum += loss.detach()
            pred = output['logits']  # [B T vocab_size]
            pred[:, :, 0] = float('-inf')  # 将索引为0的元素值修改为负无穷  
            pre_time = output['stay_time']
            pre_time[:, :, 0] = float('-inf')
            y_test = y_test.to(device)
            stay_time = stay_time.to(device)
            for b in range(B):
                _, pred_indices = torch.topk(pred[b], 100)
                _, stay_indices = torch.topk(pre_time[b], 10)
                valid_mask = y_test[b] > 0
                valid_y_val = y_test[b][valid_mask]
                valid_stay_val = stay_time[b][valid_mask]
                valid_pred_indices = pred_indices[valid_mask]
                valid_stay_indices = stay_indices[valid_mask]
                # 扩展维度以进行广播比较
                valid_y_val_expanded = valid_y_val.unsqueeze(1)  # [有效长度] -> [有效长度, 1]
                valid_stay_val_expend = valid_stay_val.unsqueeze(1)

                l = valid_y_val_expanded.size(0)
                size += l
                # 检查 top-1 准确率
                acc_time += torch.sum(valid_stay_indices[:, 0:1] == valid_stay_val_expend).item()
                mse_time += torch.abs(valid_stay_indices[:, 0:1].float() - valid_stay_val_expend.float()).sum().item()
                var += torch.abs(valid_stay_indices[:, 0:1].float() - 6.6).pow(2).sum().item()
                mean += torch.sum(valid_stay_val_expend.float())
                a1 = torch.sum(valid_pred_indices[:, 0:1] == valid_y_val_expanded).item()
                a3 = torch.sum(valid_pred_indices[:, 0:3] == valid_y_val_expanded).item()
                # 检查 top-5 准确率
                a5 = torch.sum(valid_pred_indices[:, 0:5] == valid_y_val_expanded).item()
                a10 = torch.sum(valid_pred_indices[:, 0:10] == valid_y_val_expanded).item()
                acc1 += a1
                acc3 += a3
                acc5 += a5
                acc10 += a10    

    # 聚合测试结果
    val_loss_accum = val_loss_accum / batch
    val_loss_accum = val_loss_accum.clone().detach().to(device)
    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
    val_loss_accum /= dist.get_world_size()

    var = torch.tensor(var, device=device)
    mean = mean.clone().detach().to(device)
    size = torch.tensor(size, device=device)
    acc1 = torch.tensor(acc1, device=device)
    acc3 = torch.tensor(acc3, device=device)
    acc5 = torch.tensor(acc5, device=device)
    acc10 = torch.tensor(acc10, device=device)
    acc_time = torch.tensor(acc_time, device=device)
    mse_time = torch.tensor(mse_time, device=device)

    dist.all_reduce(var, op=dist.ReduceOp.SUM)
    dist.all_reduce(mean, op=dist.ReduceOp.SUM)
    dist.all_reduce(size, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc1, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc3, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc5, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc10, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_time, op=dist.ReduceOp.SUM)
    dist.all_reduce(mse_time, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        var = var.item() / size.item()
        mean = mean.item() / size.item()
        acc1 = acc1.item() / size.item()
        acc3 = acc3.item() / size.item()
        acc5 = acc5.item() / size.item()
        acc10 = acc10.item() / size.item()
        acc_time = acc_time.item() / size.item()
        mse_time = mse_time.item() / size.item()

        with open(log_file_test, "a") as f:
            f.write(f"{val_loss_accum.item():.6f}\t{acc1:.6f}\t{acc3:.6f}\t{acc5:.6f}\t{acc10:.6f}\t{acc_time:.6f}\t{mse_time:.6f}\t{mean}\t{var}\n")   
