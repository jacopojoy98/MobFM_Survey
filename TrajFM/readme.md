# TrajFM: A Vehicle Trajectory Foundation Model for Region and Task Transferability

Implementation code of Trajectory Foundation Modal (TrajFM).

## Hands-on

Install requirements:

```bash
pip install -r requirements.txt
```

Also, you need to install PyTorch. It is a bit more complicated depending on your compute platform.

Set OS env parameters:

```bash
export SETTINGS_CACHE_DIR=/dir/to/cache/setting/files;
export MODEL_CACHE_DIR=/dir/to/cache/model/parameters;
export PRED_SAVE_DIR=/dir/to/save/predictions;
export LOG_SAVE_DIR=/dir/to/save/logs;
```

Run the main script:

```bash
python main.py -s local_test;
```

## Model Structure

![framework](./assets/framework.png)

A trajectory masking and recovery scheme is proposed to unify the generation schemes of different tasks. Essentially, TrajFM can perform masking and recovery of modalities and sub-trajectories.

## Technical Structure

The parameters and experimental settings are controlled by a JSON configuration file. `settings/local_test.json` provides an example.

The `sample` directory contains subsets of the Chengdu and Xian datasets for reference and quick debugging. The full datasets have the same file format and fields.