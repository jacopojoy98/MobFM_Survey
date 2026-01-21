from dataclasses import dataclass
import torch.nn as nn
from collections import OrderedDict
import torch
from torch.nn import functional as F
import math
import numpy as np
import inspect
from .location_encoder import Vocab_emb
from torch.nn import MultiheadAttention,Transformer

def prob(x,vocab_embedding):
    # First, we need to ensure that the dimensions of x and vocab are aligned for the dot product operation.
    x = x.unsqueeze(2)  # Shape becomes [B, T, 1, 512]
    vocab_embedding = vocab_embedding.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, loc_num, 512]
    # Calculate the dot product
    cosine_similarity = torch.matmul(x, vocab_embedding.transpose(-1, -2))  # Shape becomes [B, T, 1, loc_num]
    # Remove the extra dimension
    cosine_similarity = cosine_similarity.squeeze(2)  # Shape becomes [B, T, loc_num]
    return cosine_similarity


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Gate(nn.Module):
    def __init__(self, config,k):
        super().__init__()
        self.gate_linear = nn.Linear(config.n_embd, k)  # Outputs k weights

    def forward(self, x):
        # Use softmax to ensure the weights sum to 1
        weights = F.softmax(self.gate_linear(x), dim=-1)
        return weights
    
class Gate1(nn.Module):
    def __init__(self, config,k):
        super().__init__()
        self.gate_linear = nn.Linear(config.n_embd*2, k)  # Outputs k weights

    def forward(self, x):
        # Use softmax to ensure the weights sum to 1
        weights = F.softmax(self.gate_linear(x), dim=-1)
        return weights
    
class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        #done Changed to 4 experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(4)])  # Four experts
        self.cross_expert = Expert(config)
        # self.gate = Gate(config,k=3)  # One gate

        #todo 
        self.gate = Gate(config,k=4)  # One gate, changed to four weights
        self.gate_adapt = Gate1(config,k=2)  # gate_adapt



    def forward(self, x,time_emb):
        B_cat,_,_ = x.size()
        # b = B_cat//4

        #done
        b = B_cat//5

        poi_emb = x[:b,:,:]
        geo_emb = x[b:2*b,:,:]
        rank_emb = x[2*b:3*b,:,:]
        # cross_emb = x[3*b:,:,:]

        

        #todo-change? Just added one more expert?
        text_emb = x[3*b:4*b,:,:]
        cross_emb = x[4*b:,:,:]


        # The first three embeddings are fed into three separate experts
        expert1_output = self.experts[0](poi_emb)
        expert2_output = self.experts[1](geo_emb)
        expert3_output = self.experts[2](rank_emb)

        #todo
        expert4_output = self.experts[3](text_emb)


        w_adapt = self.gate_adapt(torch.cat((cross_emb,time_emb),dim=-1))
        # The cross embedding is fed into the gate to generate weights
        weights = self.gate(cross_emb) * w_adapt[:,:, 0].unsqueeze(-1)+ self.gate(time_emb) * w_adapt[:,:, 1].unsqueeze(-1)

        # Multiply the expert outputs by the weights
        weighted_expert1 = expert1_output * weights[:,:, 0].unsqueeze(-1)
        weighted_expert2 = expert2_output * weights[:,:, 1].unsqueeze(-1)
        weighted_expert3 = expert3_output * weights[:,:, 2].unsqueeze(-1)


        #todo
        weighted_expert4 = expert4_output * weights[:,:, 3].unsqueeze(-1)


        # Sum the weighted expert outputs and add the output of the cross embedding
        # cross_output = weighted_expert1 + weighted_expert2 + weighted_expert3 + self.cross_expert(cross_emb)
        # final_output = torch.cat((expert1_output,expert2_output,expert3_output,cross_output),dim=0)

        cross_output = weighted_expert1 + weighted_expert2 + weighted_expert3 + weighted_expert4 + self.cross_expert(cross_emb)
        final_output = torch.cat((expert1_output,expert2_output,expert3_output,expert4_output,cross_output),dim=0)

        return final_output

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttention(num_heads=config.n_head,embed_dim=config.n_embd,batch_first=True)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.smoe = MoE(config)
    
    def forward(self, x, time_emb,attn_mask,pad_mask):
        #
        # pad_mask = pad_mask.repeat(4,1)

        #done
        pad_mask = pad_mask.repeat(5,1)

        y = self.ln_1(x)
        attn_output, attn_weights = self.attn(
            y,y,y,pad_mask,True,attn_mask,True,True
        )
        x = x + attn_output
        x_ = self.smoe(self.ln_2(x),time_emb)
        x = x + x_

        return x

@dataclass
class Traj_Config:
    block_size: int = 48*3 # max seq_len
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512 # embedding dim


class Traj_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.hidden_size = config.n_embd
    
        self.transformer = nn.ModuleDict(dict(
            time_embedding = nn.Embedding(48, config.n_embd),
            stay_embedding = nn.Embedding(96,config.n_embd),
            dow_embedding = nn.Embedding(8,config.n_embd),
            lon_lat_embedding = nn.Linear(2,config.n_embd),
            poi_feature_embedding = nn.Linear(34,config.n_embd),

            #todo
            text_embedding_1 = nn.Embedding(512, config.n_embd),
            text_embedding_2 = nn.Embedding(512, config.n_embd),
            text_embedding_3 = nn.Embedding(512, config.n_embd),
            text_embedding_4 = nn.Embedding(512, config.n_embd),

            flow_rank_embedding = nn.Embedding(9,config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
            
        ))
        self.vocab_embd = Vocab_emb(config)
        self.lm_head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ts_head = nn.Linear(config.n_embd,96,bias=None)
        
        # init params
        self.apply(self._init_weights) # iterate all submodule and apply init_modules
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    "Simplified condition, as it is always valid"
    def topk_last_valid_classes(self,logits: torch.Tensor,
                                y_test: torch.Tensor,
                                k: int = 5) -> torch.Tensor:
        """
        logits: [B, N, C]
        y_test: [B, N] or [B, N, 1], >0 indicates the timestep is valid
        Returns: [B, k], the top-k class indices for each sample at the "last valid timestep"
               If a sample has no valid timesteps, that row is filled with -1
        """
        assert logits.dim() == 3, "logits must be [B, N, C]"
        B, N, C = logits.shape
    
        # Normalize mask shape to [B, N]
        mask = (y_test > 0)
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1)
        assert mask.shape == (B, N), "y_test shape must broadcast to [B, N]"
    
        # -- Find the index of the "last valid timestep" for each sample --
        # Common padding scenario (valid steps are continuous from the beginning): directly using count - 1 is faster
        lengths = mask.sum(dim=1)                      # [B]
        last_idx = (lengths - 1).clamp(min=0, max=N-1) # [B]
        no_valid = lengths == 0                        # [B], indicates none are valid
    
        # -- Get the class scores for the last valid timestep of each sample [B, C] --
        rows = torch.arange(B, device=logits.device)
        last_step_logits = logits[rows, last_idx, :]   # [B, C]
    
        # # For samples with no valid steps, set the entire row to -inf to avoid being selected by topk
        # if no_valid.any():
        #     last_step_logits = last_step_logits.masked_fill(no_valid.unsqueeze(1), float('-inf'))
    
        # -- Perform top-k on the class dimension (you can directly take the 5 you need) --
        # k = min(k, C)
        _, topk_idx = torch.topk(last_step_logits, k=k, dim=1)


        gt = y_test[rows, last_idx]
        has_gt = (topk_idx == gt.unsqueeze(1)).any(dim=1)
        need_replace  = ~has_gt
        if need_replace.any():
            gt = gt.to(device=topk_idx.device, dtype=topk_idx.dtype)
            topk_idx[need_replace, -1] = gt[need_replace]

        # For samples with no valid steps, use -1 as a sentinel index
        # if no_valid.any():
        #     topk_idx[no_valid] = -1
    
        return topk_idx,last_idx,gt  # [B, k]


    def forward(self, his,poi, ts,dow ,vocab=None,device='cuda',targets=None,target_time=None,stay_his_time = None,return_hidden_vertors=True):

        # print(self.lm_head.weight.dtype,'self.lm_head.weight.dtype')
            #   'vision_tower.dtype',)
        device = his.device
        # his ,poi,ts= his.to(device), poi.to(device),ts.to(device)
        # dow = dow.to(device)
        # if targets is not None:
        #     targets = targets.to(device)
        if target_time is not None:
            # target_time = target_time.to(device)
            stay_his_time = torch.roll(target_time,shifts=1,dims=-1)
        B, T = his.size()
        padding_mask = (his==0).to(torch.bool)
        ts = ts.to(torch.long)
        dow = dow.to(torch.long)
        stay_his_time = stay_his_time.to(torch.long)
        poi_feature = poi[:,:,:34]
        lon_lat = poi[:,:,68:70]
        #todo 
        vocab_text1 = poi[:,:, 71].to(torch.long)
        vocab_text2 = poi[:,:, 72].to(torch.long)
        vocab_text3 = poi[:,:, 73].to(torch.long)
        vocab_text4 = poi[:,:, 74].to(torch.long)


        rank = poi[:,:,70].to(torch.long)
        # vocab = vocab.to(device)
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=device) #shape (T)
        pos_emb = self.transformer.wpe(pos) 
        poi_feature_emb = self.transformer.poi_feature_embedding(poi_feature)
        lon_lat_emb = self.transformer.lon_lat_embedding(lon_lat)
        rank_emb = self.transformer.flow_rank_embedding(rank)

        # if vocab_text4[vocab_text4>=512]:
        # print("vocab_text4-forward",vocab_text4[vocab_text4>=512])
        # raise ValueError("vocab_text4 has values >= 512, check your data preprocessing.")
        #todo
        text_embedding_1  = self.transformer.text_embedding_1(vocab_text1)
        text_embedding_2  = self.transformer.text_embedding_2(vocab_text2)
        text_embedding_3  = self.transformer.text_embedding_3(vocab_text3)
        text_embedding_4  = self.transformer.text_embedding_4(vocab_text4)

        ts_emb = self.transformer.time_embedding(ts) #B T 16*3 
        stay_embd = self.transformer.stay_embedding(stay_his_time)
        dow_embd = self.transformer.dow_embedding(dow)

        time_emb = ts_emb + pos_emb + stay_embd + dow_embd

        poi_in_emb = poi_feature_emb + time_emb
        geo_in_emb = lon_lat_emb + time_emb
        rank_in_emb = rank_emb + time_emb

        #todo
        text_in_emb = text_embedding_1 + text_embedding_2 +text_embedding_3 +text_embedding_4 + time_emb
        cross_in_emb = poi_feature_emb + lon_lat_emb + rank_emb + text_embedding_1 + text_embedding_2 +text_embedding_3 +text_embedding_4

        # x = torch.cat((poi_in_emb,geo_in_emb,rank_in_emb,cross_in_emb),dim=0)
        x = torch.cat((poi_in_emb,geo_in_emb,rank_in_emb,text_in_emb,cross_in_emb),dim=0)


        all_hidden_states = ( ) if return_hidden_vertors else None

        mask = Transformer.generate_square_subsequent_mask(T,device=device).to(torch.bool)
        for block in self.transformer.h:
            x = block(x, time_emb,mask,padding_mask)

            all_hidden_states +=(x[4*B:,:,:],)
        # x = x[3*B:,:,:] 
        # if return_hidden_vertors:
        #     return all_hidden_states

        #done
        x = x[4*B:,:,:] 

        x = self.transformer.ln_f(x+time_emb)
        x = self.lm_head(x) #(b,len,dim)

        vocab_embedding = self.vocab_embd(vocab)

        # torch.save('vocab_embedding_plus_2.pt',vocab_embedding)

        logits = prob(x,vocab_embedding)



        "Optimized code for speed"
    
        topk_idx,last_idx,gt = self.topk_last_valid_classes(logits,targets,5) 


        rows = torch.arange(B, device=logits.device)
        last_x = x[rows, last_idx, :]   # [B, C]


        if return_hidden_vertors:
            return all_hidden_states,last_x,topk_idx,last_idx,gt
        




    
    def configure_optimizers(self, weight_decay, learning_rate):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} params")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params")
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        if torch.cuda.is_available():
            device = "cuda"
        use_fused = fused_available and device == "cuda" ## 8. fuse the adamw
        print(f"using fused AdamW: {use_fused}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        return optimizer