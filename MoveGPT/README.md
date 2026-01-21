# MoveGPT

A pytorch implementation for the paper: **MoveGPT: Scaling Mobility Foundation Models
with Spatially-Aware Mixture of Experts**

## ⚙️ Installation

### Environment

- Tested OS: Linux
- Python >= 3.11
- torch == 2.0.1
- CUDA == 11.7

### Dependencies:

1. Install Pytorch with the correct CUDA version.
2. Use the `pip install -r requirements.txt` command to install all of the Python modules and packages used in this project.

## ⚖ Repo Structure

```
MoveGPT 
├─location_feature                  # Location features [N,71], N means number of locations, 71=34*2(poi feature)+2(longitude & latitude)+1(popularity rank)  
│   └─vocab_{city}.npy  
│  
├─traj_dataset                      # The dataset examples where each trajectory is formatted as [user_id location,weekday,time;location,weekday,time;...].  
│       ├─test  
│       ├─train   
│       └─val   
│  
├─dataloader.py  
├─location_encoder.py          # The architecture of location encoder  
├─model.py                         # The architecture of MoveGPT  
├─utils.py                         # Train and evaluate methods  
├─train_ddp.py                     # Pretraing of multicity dataset
├─test.py                          # Test of multicity dataset
├─train_tune.py                    # Fine-tune of target city data
├─test_tune.py                     # Test of fine-tuned model
└─requirements.txt
```

## 🏃 Model Training

Pre-train MoveGPT with multi-city datasets as the following examples:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 train_ddp.py --rank 4 
```

Fine-tune MoveGPT with target city Altanda dataset :

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 train_tune.py --rank 4 
```

Test MoveGPT on target city Altanda dataset:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 test_tune.py --rank 4 
```

Once your model is trained, you will find the logs recording the training process in the  `./{args.city}` directory.
