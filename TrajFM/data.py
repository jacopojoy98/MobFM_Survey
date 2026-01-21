import math
import random

from einops import repeat
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd



TRAJ_ID_COL = 'trip'
X_COL = 'lng'
Y_COL = 'lat'
T_COL = 'timestamp'
DT_COL = 'delta_t'
ROAD_COL = 'road'
FEATURE_PAD = 0
ST_MAP = {
    "spatial": [0, 1],
    "temporal": [2, 3]
}

KNOWN_TOKEN = 0
MASK_TOKEN = 1
START_TOKEN = 2
END_TOKEN = 3
UNKNOWN_TOKEN = 4
PAD_TOKEN = 5


def coord_transform_GPS_UTM(traj, UTM_region, origin_coord = "latlong", dest_coord = "utm"):
    from pyproj import Proj, transform
    
    if origin_coord == "latlong":
        origin = Proj(proj="latlong", datum="WGS84")
        dest = Proj(proj="utm", zone=UTM_region, datum="WGS84")  # 成都位于 UTM 第48N区
        
    elif origin_coord == "utm":
        dest = Proj(proj="latlong", datum="WGS84")
        origin = Proj(proj="utm", zone=UTM_region, datum="WGS84")  # 成都位于 UTM 第48N区

    else:
        raise NotImplementedError(f'coord type error')

    if traj.ndim == 2:
        easting, northing = transform(origin, dest, traj[:,0], traj[:,1])
        traj[:,0] = easting
        traj[:,1] = northing
    elif traj.ndim == 3:
        easting, northing = transform(origin, dest, traj[:,:,0], traj[:,:,1])
        traj[:,:,0] = easting
        traj[:,:,1] = northing
    return traj

class TrajFMDataset(Dataset):
    def __init__(self, traj_df, UTM_region, scale, spatial_middle_coord=None):
        """
        Dataset supporter for the Trajectory Foundation Model.

        Args:
            traj_df (pd.DataFrame): contains points of all trajectories.
        """
        super().__init__()

        self.traj_df = traj_df
        
        self.UTM_region = UTM_region
        self.scale = scale
        self.traj_df['timestamp'] = self.traj_df['time'].apply(lambda x: x.timestamp())
        self.traj_ids = self.traj_df[TRAJ_ID_COL].unique()

        spatial_border = traj_df[[X_COL, Y_COL]]
        self.spatial_border = [spatial_border.min().tolist(), spatial_border.max().tolist()]
        
        if spatial_middle_coord is None:
            self.middle_coord = np.array([[(self.spatial_border[0][0] + self.spatial_border[1][0])/2, (self.spatial_border[0][1] + self.spatial_border[1][1])/2]]) #中心点经纬度坐标
            self.spatial_middle_coord = coord_transform_GPS_UTM(self.middle_coord, self.UTM_region)
        else:
            self.spatial_middle_coord = spatial_middle_coord
        # 进行缩放操作
        traj_gps = traj_df[[X_COL, Y_COL]].values
        traj_utm = (coord_transform_GPS_UTM(traj_gps, self.UTM_region) - self.spatial_middle_coord) / self.scale 
        
        self.traj_df[[X_COL, Y_COL]] = pd.DataFrame(traj_utm)
        

    def __len__(self):
        return self.traj_ids.shape[0]

    def __getitem__(self, index):
        one_traj = self.traj_df[self.traj_df[TRAJ_ID_COL] == self.traj_ids[index]].copy()
        one_traj[DT_COL] = one_traj[T_COL] - one_traj[T_COL].iloc[0]
        return one_traj

class PretrainPadder:
    """A Pre-training padder made by mix the input and output pairs of all three types of downstream tasks.
    """
    def __init__(self, tp_pred_len, mix_ratio=[1, 1, 1]):
        _total = sum(mix_ratio)
        normalized_ratios = [ratio / _total for ratio in mix_ratio]
        self.mix_ratio = normalized_ratios

        self.task_padders = [TpPadder(tp_pred_len, eval=False),
                             TrajTtePadder(),
                             OdTtePadder(eval=False)]

    def __call__(self, raw_batch):
        selected_case = random.choices(range(3), weights=self.mix_ratio, k=1)[0]
        return self.task_padders[selected_case](raw_batch)


class PretrainPadder:
    """Collate function for the pre-training.
    """

    def __init__(self, span_div_ratio, span_mask_ratio, feature_mask_prob):
        self.span_div_ratio = span_div_ratio
        self.span_mask_ratio = span_mask_ratio
        self.feature_mask_prob = feature_mask_prob

    def __call__(self, raw_batch):
        """
        A function for padding the provided raw batch into a standard array.

        Features:
            0 - Longitude
            1 - Latitude
            2 - Timestamp
            3 - Timestamp delta
        Every feature is 2D: the first dimension is the value, the second is the token.

        Args:
            raw_batch (list): each item is a pd.DataFrame representing one trajectory.

        Returns:
            np.array: the padded input array, with shape (B, L, F, 2)
            np.array: the padded output array, with shape (B, L, F, 2)
            np.array: the padded dual-layer positions, with shape (B, L, 2)
        """
        input_batch, output_batch, pos_batch = [], [], []
        for traj in raw_batch:
            prompt_arr, src_arr, trg_arr, prompt_pos, gen_pos = [], [], [], [], []
            prompt = traj[[X_COL, Y_COL, T_COL, DT_COL]].to_numpy()
            valid_len = prompt.shape[0]

            span_div_idx = sorted(set(np.random.choice(valid_len, math.ceil(valid_len * self.span_div_ratio), replace=False)) |
                                  set([0, valid_len]))
            spans_idx = np.stack([span_div_idx[:-1], span_div_idx[1:]], axis=1)
            masked_span_i = np.random.choice(spans_idx.shape[0], math.ceil(
                spans_idx.shape[0] * self.span_mask_ratio), replace=False)

            cur_pos = 0
            for i, (l, r) in enumerate(spans_idx):
                span = prompt[l:r]
                if i in masked_span_i:
                    mask_values = np.ones([1, span.shape[-1]]) * FEATURE_PAD  # Mask the span with FEATURE_PAD
                    # Add mask tokens to input
                    prompt_arr.append(np.stack((mask_values, np.full((1, span.shape[-1]), MASK_TOKEN)), axis=-1))
                    prompt_pos.append(np.stack((np.full(1, cur_pos), np.zeros(1)), axis=-1))
                    # Source spans and target spans
                    src_arr.append(np.concatenate((
                        np.stack((np.full((1, span.shape[-1]), FEATURE_PAD),
                                 np.full((1, span.shape[-1]), START_TOKEN)), axis=-1),
                        np.stack((span, np.full((r-l, span.shape[1]), UNKNOWN_TOKEN)), axis=-1)
                    )))
                    trg_arr.append(np.concatenate((
                        np.stack((span, np.full((r-l, span.shape[1]), UNKNOWN_TOKEN)), axis=-1),
                        np.stack((np.full((1, span.shape[-1]), FEATURE_PAD),
                                 np.full((1, span.shape[-1]), END_TOKEN)), axis=-1)
                    )))
                    gen_pos.append(
                        np.stack((np.full(r - l + 1, cur_pos), np.arange(1, r - l + 2)), axis=-1)
                    )
                    cur_pos += 1
                else:
                    # Known span
                    prompt_arr.append(np.stack((span, np.full((r-l, prompt.shape[1]), KNOWN_TOKEN)), axis=-1))
                    prompt_pos.append(
                        np.stack((np.arange(cur_pos, cur_pos + r - l), np.zeros(r - l)), axis=-1)
                    )
                    cur_pos += r - l

            prompt_arr = np.concatenate(prompt_arr, axis=0)
            src_arr = np.concatenate(src_arr, axis=0)
            trg_arr = np.concatenate(trg_arr, axis=0)
            pos_arr = np.concatenate((np.concatenate(prompt_pos, axis=0), np.concatenate(gen_pos, axis=0)), axis=0)
            # Mask prompt features
            # Fill the masked input tokens with MASK_TOKEN, and the masked output tokens with UNKNOWN_TOKEN
            # Avoid masking the spatial and temporal features simultaneously
            prompt_mask = np.random.rand(prompt_arr.shape[0]) < self.feature_mask_prob
            spatial_mask = np.random.rand(prompt_arr.shape[0]) < 0.5
            temporal_mask = ~spatial_mask
            spatial_mask = repeat(prompt_mask & spatial_mask, 'L -> L F', F=prompt_arr.shape[1])
            spatial_mask[:, ST_MAP['temporal']] = False
            temporal_mask = repeat(prompt_mask & temporal_mask, 'L -> L F', F=prompt_arr.shape[1])
            temporal_mask[:, ST_MAP['spatial']] = False

            input_prompt = prompt_arr
            ouput_prompt = np.copy(prompt_arr)
            input_prompt[spatial_mask, 0] = FEATURE_PAD
            input_prompt[spatial_mask, 1] = MASK_TOKEN
            input_prompt[temporal_mask, 0] = FEATURE_PAD
            input_prompt[temporal_mask, 1] = MASK_TOKEN
            ouput_prompt[spatial_mask, 1] = UNKNOWN_TOKEN
            ouput_prompt[temporal_mask, 1] = UNKNOWN_TOKEN

            input_batch.append(np.concatenate((input_prompt, src_arr), axis=0))
            output_batch.append(np.concatenate((ouput_prompt, trg_arr), axis=0))
            pos_batch.append(pos_arr)

        # Pad the input and output arrays
        input_batch = torch.tensor(pad_batch_3d(input_batch)).float()
        output_batch = torch.tensor(pad_batch_3d(output_batch)).float()
        pos_batch = torch.tensor(pad_batch_2d(pos_batch)).long()

        return input_batch, output_batch, pos_batch

        
class TpPadder:
    """Collate function for the Trajectory Prediction (TP) task.
    """

    def __init__(self, pred_len, eval=True):
        self.pred_len = pred_len
        self.eval = eval

    def __call__(self, raw_batch):
        """Padding the provided raw batch into trajectory feature tensors.
        Refer to PretrainPadder for detailed definition on the features.
        """
        input_batch, output_batch, pos_batch = [], [], []
        for traj in raw_batch:
            traj_feats = traj[[X_COL, Y_COL, T_COL, DT_COL]].to_numpy()  # (L, F)

            input_row = traj_feats[:-self.pred_len]
            input_row = np.stack([input_row, np.ones_like(input_row) * KNOWN_TOKEN], -1)  # (L, F, 2)
            input_row = np.concatenate([input_row, repeat(np.array([FEATURE_PAD, MASK_TOKEN]),
                                                          'a -> 1 F a', F=input_row.shape[1])])
            input_batch.append(input_row)

            output_row = traj_feats[-self.pred_len:]
            output_row = np.stack([output_row, np.ones_like(output_row) * UNKNOWN_TOKEN], -1)
            output_batch.append(output_row)

            pos_batch.append(np.stack([np.arange(input_row.shape[0]), np.zeros((input_row.shape[0]))], -1))  # (L, 2)

        input_batch, output_batch, pos_batch = pad_batch_3d(input_batch), \
            pad_batch_3d(output_batch), pad_batch_2d(pos_batch)  # (B, L_in/out, F, 2), (B, L_in, 2)

        input_tensor = torch.from_numpy(
            np.concatenate([input_batch,
                            repeat(np.array([FEATURE_PAD, START_TOKEN]), 'a -> B 1 F a',
                                   B=input_batch.shape[0], F=input_batch.shape[2])], axis=1)).float()
        if not self.eval:
            input_tensor = torch.cat([input_tensor, torch.from_numpy(output_batch).float()], dim=1)
        output_tensor = torch.from_numpy(
            np.concatenate([input_batch, output_batch,
                            repeat(np.array([FEATURE_PAD, END_TOKEN]), 'a -> B 1 F a',
                                   B=input_batch.shape[0], F=input_batch.shape[2])], axis=1)).float()
        pos_tensor = torch.from_numpy(
            np.concatenate([pos_batch,
                            np.stack([repeat(np.max(pos_batch, axis=1)[..., 0], 'B -> B L', L=self.pred_len+1),
                                      repeat(np.arange(1, self.pred_len+2), 'L -> B L', B=input_batch.shape[0])],
                                     axis=-1)], axis=1)).long()

        return input_tensor, output_tensor, pos_tensor


class TrajTtePadder:
    """Collate function for the Trajectory-based TTE Task.
    """

    def __init__(self):
        pass

    def __call__(self, raw_batch):
        """Padding the provided raw batch into trajectory feature tensors.
        Refer to PretrainPadder for detailed definition on the features.
        """
        input_batch, output_batch, pos_batch = [], [], []
        for traj in raw_batch:
            input_row = traj[[X_COL, Y_COL, T_COL, DT_COL]].to_numpy()  # (L, F)
            
            input_row = np.stack([input_row, np.ones_like(input_row) * KNOWN_TOKEN], -1)  # (L, F, 2)
            output_row = np.copy(input_row)
            pos_row = np.stack([np.arange(input_row.shape[0]), np.zeros((input_row.shape[0]))], -1)

            input_row[1:, ST_MAP['temporal'], 0] = FEATURE_PAD
            input_row[1:, ST_MAP['temporal'], 1] = MASK_TOKEN
            output_row[1:, ST_MAP['temporal'], 1] = UNKNOWN_TOKEN

            input_batch.append(input_row)
            output_batch.append(output_row)
            pos_batch.append(pos_row)  # (L, 2)

        input_batch = torch.tensor(pad_batch_3d(input_batch)).float()
        output_batch = torch.tensor(pad_batch_3d(output_batch)).float()
        pos_batch = torch.tensor(pad_batch_2d(pos_batch)).long()
        return input_batch, output_batch, pos_batch


class OdTtePadder:
    """Collate function for the OD-based TTE task.
    """

    def __init__(self, eval=True):
        self.eval = eval

    def __call__(self, raw_batch):
        """Padding the provided raw batch into trajectory feature tensors.
        Refer to PretrainPadder for detailed definition on the features.
        """
        input_batch, output_batch, pos_batch = [], [], []
        for traj in raw_batch:
            traj_feats = traj[[X_COL, Y_COL, T_COL, DT_COL]].to_numpy()  # (L, F)
            input_row = traj_feats[[0, 1, -1]]
            input_row = np.stack([input_row, np.ones_like(input_row) * KNOWN_TOKEN], -1)  # (L, F, 2)

            input_row[1, :, 0] = FEATURE_PAD
            input_row[1, :, 1] = MASK_TOKEN
            output_row = np.copy(input_row)
            input_row[2, ST_MAP['temporal'], 0] = FEATURE_PAD
            input_row[2, ST_MAP['temporal'], 1] = MASK_TOKEN
            output_row[2, ST_MAP['temporal'], 1] = UNKNOWN_TOKEN
            pos_row = np.array([[0, 0], [1, 0], [2, 0]])

            if not self.eval:
                sub_traj = traj_feats[1:-1]
                sub_traj = np.stack([sub_traj, np.ones_like(sub_traj) * UNKNOWN_TOKEN], -1)  # (L, F, 2)
                input_row = np.concatenate([
                    input_row, repeat(np.array([FEATURE_PAD, START_TOKEN]), 'a -> 1 F a', F=sub_traj.shape[1]),
                    sub_traj
                ], axis=0)
                output_row = np.concatenate([
                    output_row, sub_traj, repeat(np.array([FEATURE_PAD, END_TOKEN]), 'a -> 1 F a', F=sub_traj.shape[1])
                ], axis=0)
                pos_row = np.concatenate([
                    pos_row, np.stack([np.ones(sub_traj.shape[0] + 1) * 1, np.arange(1, sub_traj.shape[0] + 2)], -1)
                ], axis=0)

            input_batch.append(input_row)
            output_batch.append(output_row)
            pos_batch.append(pos_row)

        input_batch = torch.tensor(pad_batch_3d(input_batch)).float()
        output_batch = torch.tensor(pad_batch_3d(output_batch)).float()
        pos_batch = torch.tensor(pad_batch_2d(pos_batch)).long()
        
        return input_batch, output_batch, pos_batch


def fetch_task_padder(padder_name, padder_params):
    if padder_name == 'tp':
        task_padder = TpPadder(**padder_params)
    elif padder_name == 'traj_tte':
        task_padder = TrajTtePadder()
    elif padder_name == 'od_tte':
        task_padder = OdTtePadder(**padder_params)
    else:
        raise NotImplementedError(f'No Padder named {padder_name}')

    return task_padder


def pad_batch_3d(batch):
    """
    Pad the batch to the maximum length of the batch.

    Args:
        batch (list): the batch of arrays to pad, [(L1, F, 2), (L2, F, 2), ...].

    Returns:
        np.array: the padded array.
    """
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch = np.stack((
        np.full((len(batch), max_len, batch[0].shape[1]), FEATURE_PAD, dtype=float),
        np.full((len(batch), max_len, batch[0].shape[1]), PAD_TOKEN, dtype=float)
    ), axis=-1)
    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr

    return padded_batch


def pad_batch_2d(batch):
    """
    Pad the batch to the maximum length of the batch.

    Args:
        batch (list): the batch of arrays to pad, [(L1, 2), (L2, 2), ...].

    Returns:
        np.array: the padded array.
    """
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch = np.stack((
        np.full((len(batch), max_len), FEATURE_PAD, dtype=float),
        np.full((len(batch), max_len), FEATURE_PAD, dtype=float)
    ), axis=-1)
    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr

    return padded_batch