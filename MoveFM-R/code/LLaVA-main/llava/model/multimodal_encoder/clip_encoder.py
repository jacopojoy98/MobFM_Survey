import torch
import torch.nn as nn

from typing import Optional
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
# import flash_attn
from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import PretrainedConfig
from transformers import AutoModel
from .GPT_model_seq import *
# data_loader
from torch.utils.data import Dataset, DataLoader
import os
from torch.nn import functional as F
import random
from .traj_moe  import Traj_Config, Traj_Model



# vocab_city_dict = {}
vocab_tot = None
for city in  ['Atlanta','Chicago','LosAngeles','NewYork','Seattle','WashingtonDC']:
    vocab = np.load(f'location_feature/vocab_{city}.npy')
    # 
    vocab = np.pad(vocab, ((2, 0), (0, 0)), mode='constant', constant_values=0)
    # Convert to tensor
    vocab = torch.from_numpy(vocab).float().cuda()
    # vocab_city_dict[len(vocab_city_dict)] = vocab.shape[0] 
    if vocab_tot is None:
        vocab_tot = vocab +1 -1 
    else:
        vocab_tot = torch.cat([vocab_tot, vocab], dim=0)


class TransformerBlock(nn.Module):
    def __init__(self,d_model, d_ff, num_head, add_LoRA = False, LoRA_dim = 8,use_rope=False):
        super(TransformerBlock, self,).__init__()
        if num_head == 1:
            self.attention = LinearAttention(d_model)
        else:
            self.attention = MultiHeadLinearAttention(d_model, num_head,use_rope)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.add_LoRA = add_LoRA
        if self.add_LoRA:
            self.lora = LoRA_layer(d_model, LoRA_dim)

    def forward(self,x):
        if self.add_LoRA:
            x_lora = self.lora(x)
        attention_output = self.attention(x)
        x_out = self.norm1(x + attention_output)

        feed_forward_output = self.feed_forward(x_out)
        y = self.norm2(x+ feed_forward_output)
        if self.add_LoRA:
            y = y + x_lora
        return y


class PositionalEncoder(nn.Module):
    def __init__(self, dropout = 0.1, max_seq_len = 100, d_model = 32, batch_first = True):
        super( ).__init__( )
        self.dropout = nn.Dropout(p = dropout)
        self.max_seq_len = max_seq_len
        self.batch_first = batch_first
        self.d_model = d_model

        self.x_dim = 1 if batch_first else 0
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe.squeeze(1)[:x.shape[self.x_dim]]
        return self.dropout(x)


class FlashTransformer(nn.Module):
    def __init__(self, d_model, d_ff, num_head, num_layer,add_adapter = False, num_adapter = 3, add_lora = False, lora_dim = 8,use_rope=False):
        super(FlashTransformer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_head = num_head
        self.add_adapter = add_adapter
        self.num_adapter = num_adapter
        self.num_layer = num_layer
        self.per_layers = num_layer // num_adapter
        if add_adapter:
            assert num_layer % num_adapter == 0, 'num_layer should be divisible by num_adapter'
        self.per_num = [i for i in range(num_layer) if i % self.per_layers == (self.per_layers - 1)]
        self.layers = nn.ModuleList([TransformerBlock(d_model, d_ff, num_head, add_lora, lora_dim,use_rope) for _ in range(num_layer)])
        if add_adapter:
            self.adapter = nn.ModuleList([nn.Linear(d_model, d_model) for i in range(num_adapter)])
        

    def forward(self, x,hidden_states=True):
        all_hidden_states = ( ) if hidden_states else None
        for i in range(self.num_layer):
            x= self.layers[i](x)
          
            if self.add_adapter and i in self.per_num:
                x = self.adapter[i // self.per_layers](x)
            all_hidden_states +=(x,)
        if hidden_states:
            return all_hidden_states
        return x


class GPTModel_sample_autoloss_flash(nn.Module):

    def __init__(self, n_locs,n_events, dim, seq_length,pred_length, n_layers, pred_event, pred_intent, n_heads, config = None, use_rope=False ):
        super(GPTModel_sample_autoloss_flash, self).__init__( )

        self.n_locs = n_locs
        self.n_events = n_events
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.dim = dim
        self.n_layers = n_layers
        self.pred_event = pred_event
        self.pred_intent = pred_intent
        self.n_head = n_heads
        self.config = config

        self.week_emb = nn.Embedding(7, dim )
        self.hour_emb = nn.Embedding(96, dim )
        self.loc_emb = nn.Embedding(n_locs, dim)
        self.event_emb = nn.Embedding(n_events, dim)
        
        self.pos_encoder = PositionalEncoder(dropout = 0.1, max_seq_len = self.seq_length, d_model = self.dim *4)

        self.gpt2 = FlashTransformer(dim * 4, 3072, n_heads,n_layers,use_rope=use_rope)
        
       
        self.classifier_1 = nn.Linear(dim * 4, dim *3 )
        self.classifier_2 = nn.Linear(dim * 3, dim)
        self.output_layer = nn.Linear(dim, pred_intent)

        self.classifier_event = nn.Linear(dim * 4, pred_event)

        self.hidden_size = self.dim * 4

    def random_mask(self,tensor, mask_percent = 0.25):
        batch, length, dim = tensor.shape
        num_masked = int(mask_percent * length)

        mask = torch.zeros(length, dtype = torch.bool)
        mask_index = torch.randperm(length)[:num_masked]
        mask[mask_index] = True
        tensor_masked = tensor.clone()
        tensor_masked[:,mask,:] = 0
        return tensor_masked, mask 


    def forward(self,  his_weekday, his_timestamp, his_loc,his_event,pred_weekday, pred_timestamp, pred_loc,output_hidden_states=False):


        his_weekday = self.week_emb(his_weekday)
        his_timestamp = self.hour_emb(his_timestamp)

        his_loc = self.loc_emb(his_loc)

        his_event = self.event_emb(his_event)

        pred_weekday = self.week_emb(pred_weekday)
        pred_timestamp = self.hour_emb(pred_timestamp)
        pred_loc = self.loc_emb(pred_loc)


        his_emb = torch.cat([  his_weekday,his_timestamp,his_loc, his_event],dim = -1)
        pred_emb = torch.cat([  pred_weekday,pred_timestamp,pred_loc,],dim = -1)

        enc_emb = self.pos_encoder(his_emb)
        

        enc_emb = self.gpt2(enc_emb, )

        if output_hidden_states:
            return enc_emb

        enc_emb = torch.mean(enc_emb, 1).unsqueeze(1)
        enc_emb = torch.sigmoid(self.classifier_1(enc_emb)) *pred_emb
        enc_emb = enc_emb

        out = self.classifier_2(enc_emb) 
        out = torch.sigmoid(out)
        out = self.output_layer(out)

        masked_his_emb, mask = self.random_mask(his_emb)
        masked_enc_emb = self.gpt2( self.pos_encoder(masked_his_emb))
        event = self.classifier_event(masked_enc_emb)

        return out, event, mask



class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            # self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return



        if "traj" in self.vision_tower_name:

            traj_n_embd = 512
            traj_n_head = 4
            traj_n_layer = 4


            self.vision_tower = Traj_Model(Traj_Config(n_embd=traj_n_embd, n_head=traj_n_head, n_layer=traj_n_layer)).cuda()


        state_dcit = torch.load(self.vision_tower_name)

        new_state_dict ={}
        for k,v in state_dcit.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v 

        self.vision_tower.load_state_dict(new_state_dict)



        self.vision_tower.requires_grad_(False)

        self.is_loaded = True


    def feature_select(self, image_forward_outs):

        image_features = image_forward_outs[self.select_layer]

        return image_features

    @torch.no_grad()
    def forward(self, images,model_dtype):
        if type(images) is list:
            print('llava/model/multimodal_encoder/clip_encoder.py  this is wrong')
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out)
                image_features.append(image_feature)
        else:
            
            #traj
            traj = images
            # print(images.shape)
            his_len = traj.shape[1]-1
            x_test = traj[:,:-1, 0].int().reshape(-1,his_len)
            y_test = traj[:,1:, 0].int().reshape(-1,his_len)


            ts_his = traj[:,:-1, 1].int().reshape(-1,his_len)
            ts_next = traj[:,1:,1].int().reshape(-1,his_len)
            day_his = traj[:,:-1,2].int().reshape(-1,his_len)
            day_next = traj[:,1:,2].int().reshape(-1,his_len)

            #
            test_city = traj[:,:-1,3].int().reshape(-1,his_len)

            device = traj.device

            condition = day_his == day_next
            condition = ~condition
            zero_matrix = day_his * 0
            result_weekday = torch.where(condition, 48, zero_matrix).to(device)
            stay_time = result_weekday + ts_next - ts_his

            
            vocab_tot_device = vocab_tot.device 


            "Keep precision at fp32"
            poi = vocab_tot[x_test.to(vocab_tot_device)+test_city.to(vocab_tot_device)].to(device)


            image_forward_outs,last_x,topk_idx,last_idx,gt = self.vision_tower(x_test,poi, ts_his, day_his, vocab_tot.to(device), device, y_test, stay_time)

            image_features = self.feature_select(image_forward_outs)

        return image_features,last_x,topk_idx,last_idx,gt

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.vision_tower.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            print('config llava_ini/llava/model/multimodal_encoder/clip_encoder.py - this is wrong')
            pr(123)
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

    @property
    def num_patches_per_side(self):
        print('no this is wrong')
        pr(123)
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        print('no llava_ini/llava/model/multimodal_encoder/clip_encoder.py - num_patches')
        pr(123)
        return (self.config.image_size // self.config.patch_size) ** 2