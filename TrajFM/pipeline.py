from time import time

import pandas as pd
import numpy as np
import torch
from tqdm import trange, tqdm


def train_model(model, dataloader, device, num_epoch, lr):
    """Train the model given the training dataloader.

    Args:
        model (nn.Module): the model to train.
        dataloader (DataLoader): batch iterator containing the training data.
        num_epoch (int): number of epoches to train.
        lr (float): learning rate for the optimizer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    bar_desc = 'Training, avg loss: %.5f'
    log = []
    saved_model_state_dict = None
    best_loss = 1e9
    with trange(num_epoch, desc=bar_desc % 0.0) as bar:
        for epoch_i in bar:
            loss_values = []
            epoch_time = 0
            
            for batch in tqdm(dataloader, desc='-->Traversing', leave=False):
                (input_tensor, output_tensor, pos_tensor) = batch
                
                input_tensor, output_tensor, pos_tensor = input_tensor.to(device), output_tensor.to(device), pos_tensor.to(device)
                optimizer.zero_grad()
                s_time = time()
                loss = model.loss(input_tensor, output_tensor, pos_tensor)
                loss.backward()
                optimizer.step()
                e_time = time()
                loss_values.append(loss.item())
                epoch_time += e_time - s_time
            loss_epoch = np.mean(loss_values)
            bar.set_description(bar_desc % loss_epoch)
            log.append([epoch_i, epoch_time, loss_epoch])

            if loss_epoch < best_loss:
                best_loss = loss_epoch
                saved_model_state_dict = model.state_dict()
                
    log = pd.DataFrame(log, columns=['epoch', 'time', 'loss'])
    log = log.set_index('epoch')
    return log, saved_model_state_dict


@torch.no_grad()
def test_model(model, device, dataloader):
    """Test the model given the testing dataloader.

    Args:
        model (nn.Module): the model to test.
        dataloader (dataloader): batch iterator containing the testing data.
    """
    model.eval()

    predictions, targets = [], []
    for batch in tqdm(dataloader, desc='Testing'):
        (input_tensor, output_tensor, pos_tensor) = batch
        input_tensor, output_tensor, pos_tensor = input_tensor.to(device), output_tensor.to(device), pos_tensor.to(device)
        pred, target = model.test(input_tensor, output_tensor, pos_tensor)
        predictions.append([p for p in pred])
        targets.append([t for t in target])

    predictions = [pad_batch_arrays(item) for item in zip(*predictions)]
    targets = [pad_batch_arrays(item) for item in zip(*targets)]
    
    return predictions, targets


def pad_batch_arrays(arrs):
    """Pad a batch of arrays with representing feature sequences of different lengths.

    Args:
        arrs (list): each item is an array with shape (B, L, ...). The length L is different for different arrays.

    Returns:
        np.array: padded arrays with shape (B_agg, L_max, ...) that are concatenated along the batch axis.
    """
    max_len = max(a.shape[1] for a in arrs)
    arrs = [
        np.concatenate([a, np.repeat(a[:, -1:], repeats=max_len-a.shape[1], axis=1)], axis=1)
        for a in arrs
    ]
    return np.concatenate(arrs, 0)
