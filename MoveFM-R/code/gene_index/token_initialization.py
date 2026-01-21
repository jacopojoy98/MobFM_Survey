import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Set the GPU device to use

from setproctitle import setproctitle

# Read the corresponding data
np_data= np.load('area_usa_all_city_100wan_tot.npy')
text_data = json.load(open('location_best_loss_100wan_all_city_best_epoch20.index.json', 'r', encoding='utf-8'))



# ====================== Configuration ======================
MODEL_ID = "Qwen/Qwen2___5-7B-Instruct"          # or a local path

OUTPUT_DIR = "llm_models/qwen2_5_7b_with_new_vocab_100wan_code512"

BATCH_SIZE = 1024
EPOCHS = 6
LR = 2e-2
WEIGHT_DECAY = 0.01
COSINE_MARGIN = 0.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Regularization weights (do not change with batch size)
LAMBDA_PRIOR = 1e-2
LAMBDA_COH = 1e-2
TOPK_NEIGHBORS = 8
WINDOW_SIZE = 8
COH_MAX_SAMPLES = len(text_data)   

# DataLoader parallelization
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


text_data_final_data = []
for key,value in text_data.items():
    text_data_final_data.append(value)


prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>"]

set_new_token_list = []
for item in prefix:
    for i in range(512):
        set_new_token_list.append(item.format(i))


# ====================== 0) Load original tokenizer & model (get d_model & old subword embeddings) ======================
print("Loading original tokenizer & model...")
tokenizer_orig = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()
embed_module = model.get_input_embeddings()
d_model = embed_module.embedding_dim
base_weight = embed_module.weight
base_device = base_weight.device
base_dtype = base_weight.dtype
print(f"d_model = {d_model}, base_weight device = {base_device}, dtype = {base_dtype}")

# ====================== 1) Collect list of new tokens (e.g., "<a_190>") ======================
new_token_strings: List[str] = []

new_token_strings = set_new_token_list

print(f"Collected {len(new_token_strings)} new tokens (e.g., {new_token_strings[:5]}).")

# ====================== 2) Clone tokenizer and add new tokens (as non-special tokens) ======================
print("Cloning tokenizer and adding new tokens...")
tokenizer_new = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
num_added = tokenizer_new.add_tokens(new_token_strings, special_tokens=False)
assert num_added == len(new_token_strings), "Not all tokens were added due to duplicates or invalid tokens"

# Extend model's vocabulary
model.resize_token_embeddings(len(tokenizer_new))
new_token_to_id: Dict[str, int] = {t: tokenizer_new.convert_tokens_to_ids(t) for t in new_token_strings}
new_token_ids = [new_token_to_id[t] for t in new_token_strings]

# ====================== 3) Baseline initialization (mean of old subword components) ======================
@torch.no_grad()
def composition_init_vec(token_str: str) -> torch.Tensor:
    ids = tokenizer_orig(token_str, add_special_tokens=False).input_ids
    if len(ids) == 0:
        return torch.randn(d_model, dtype=base_dtype, device=base_device) * 0.02
    vecs = base_weight[torch.tensor(ids, device=base_device)]
    return vecs.mean(dim=0)

print("Computing composition-based priors...")
with torch.no_grad():
    prior_init = torch.stack([composition_init_vec(t) for t in new_token_strings])  # [num_new, d_model]
    model.get_input_embeddings().weight[torch.tensor(new_token_ids, device=base_device)] = prior_init.to(base_dtype)

# ====================== 4) Load all data into memory (no random access / binary index) ======================
@dataclass
class Item:
    seq_local: List[int]
    target: torch.Tensor

class InMemoryDataset(Dataset):
    def __init__(self, text_data_final_data, np_data,token2local_index):
        self.items: List[Item] = []
        self._load_all(text_data_final_data, np_data, token2local_index)

    def _load_all(self, text_data_final_data, np_data, token2local_index):
        d_enc = None
        for id in range(len(text_data_final_data)):
            # ex = json.loads(line)
            ex = text_data_final_data[id]
            seq = [token2local_index[t] for t in ex if t in token2local_index]
            if not seq:
                print('empty sequence found-----')
                print(ex)
                print(ex[0])
                pr(123)
                continue
            target = torch.tensor(np_data[id], dtype=torch.float32)
            if d_enc is None:
                d_enc = target.numel()
            self.items.append(Item(seq_local=seq, target=target))

        assert d_enc is not None, "Could not infer target vector dimension d_enc"
        self.d_enc = d_enc
        print(f"Loaded {len(self.items)} samples, d_enc = {self.d_enc}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Item:
        return self.items[idx]

# Local index mapping (0..num_new-1)
local_index2token = {i: t for i, t in enumerate(new_token_strings)}
token2local_index = {t: i for i, t in enumerate(new_token_strings)}
local_index2global_id = {i: new_token_to_id[t] for i, t in enumerate(new_token_strings)}

num_new = len(new_token_strings)

dataset = InMemoryDataset(text_data_final_data, np_data,token2local_index)
d_enc = dataset.d_enc

def collate_fn(batch: List[Item]):
    # Filter out empty sequences (in rare cases)
    batch = [b for b in batch if len(b.seq_local) > 0]
    if len(batch) == 0:
        return None
    lens = [len(b.seq_local) for b in batch]
    targets = torch.stack([b.target for b in batch], dim=0)  # [B, d_enc]
    flat_idx, seg = [], []
    for i, b in enumerate(batch):
        flat_idx.extend(b.seq_local)
        seg.extend([i] * len(b.seq_local))
    flat_idx = torch.tensor(flat_idx, dtype=torch.long)
    seg = torch.tensor(seg, dtype=torch.long)
    lens = torch.tensor(lens, dtype=torch.long)
    return flat_idx, seg, lens, targets

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    drop_last=True
)

# ====================== 5) (Optional) Co-occurrence graph PMI Top-K (directly calculated from in-memory data) ======================
def build_cooccurrence_from_memory(items: List[Item], window: int, topk: int, max_samples: int) -> Dict[int, List[Tuple[int, float]]]:
    token_counts = Counter()
    pair_counts = Counter()
    total_tokens = 0
    use_n = min(max_samples, len(items))
    for k in tqdm(range(use_n), desc="Building co-occurrence (PMI)"):
        seq = items[k].seq_local
        for i, ti in enumerate(seq):
            token_counts[ti] += 1
            total_tokens += 1
            j_end = min(len(seq), i + window + 1)
            for j in range(i + 1, j_end):
                tj = seq[j]
                if ti == tj:
                    continue
                a, b = (ti, tj) if ti < tj else (tj, ti)
                pair_counts[(a, b)] += 1

    if total_tokens == 0:
        return {}

    N = total_tokens
    neighbors = defaultdict(list)
    for (a, b), c in pair_counts.items():
        p_ab = c / N
        p_a = token_counts[a] / N
        p_b = token_counts[b] / N
        pmi = math.log(max(p_ab, 1e-12) / max(p_a * p_b, 1e-12))
        if pmi <= 0:
            continue
        neighbors[a].append((b, pmi))
        neighbors[b].append((a, pmi))

    W = {}
    for t, lst in neighbors.items():
        lst.sort(key=lambda x: -x[1])
        lst = lst[:topk]
        s = sum(w for _, w in lst) + 1e-9
        W[t] = [(u, w / s) for (u, w) in lst]
    return W

W_neighbors = build_cooccurrence_from_memory(dataset.items, WINDOW_SIZE, TOPK_NEIGHBORS, COH_MAX_SAMPLES) if LAMBDA_COH > 0 else {}

# ====================== 6) Alignment model (learn only new token embeddings + linear projection) ======================
class Aligner(nn.Module):
    def __init__(self, num_new: int, d_model: int, d_enc: int, prior_init: torch.Tensor):
        super().__init__()
        self.new_embed = nn.Embedding(num_new, d_model)
        with torch.no_grad():
            self.new_embed.weight.copy_(prior_init.float())
        self.proj = nn.Linear(d_model, d_enc, bias=True)
        self.register_buffer("prior", prior_init.float())

    def forward(self, flat_idx: torch.Tensor, seg: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        Vt = self.new_embed(flat_idx)     # [T, d_model]
        B = lens.shape[0]
        d_model = Vt.shape[1]
        sums = torch.zeros(B, d_model, device=Vt.device, dtype=Vt.dtype)
        sums.index_add_(0, seg, Vt)       # Summation segmented by sample
        means = sums / lens.unsqueeze(1)  # [B, d_model]
        return self.proj(means)           # [B, d_enc]

    def prior_reg(self) -> torch.Tensor:
        return torch.mean((self.new_embed.weight - self.prior) ** 2)

    def coh_reg(self, W_neighbors: Dict[int, List[Tuple[int, float]]]) -> torch.Tensor:
        if not W_neighbors:
            return torch.tensor(0.0, device=self.new_embed.weight.device)
        V = self.new_embed.weight
        loss = 0.0
        cnt = 0
        for t, lst in W_neighbors.items():
            vt = V[t]
            for u, w in lst:
                vu = V[u]
                loss = loss + w * torch.mean((vt - vu) ** 2)
                cnt += 1
        if cnt == 0:
            return torch.tensor(0.0, device=V.device)
        return loss / cnt

aligner = Aligner(num_new=num_new, d_model=d_model, d_enc=d_enc, prior_init=prior_init.cpu()).to(DEVICE)

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
def cosine_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = COSINE_MARGIN) -> torch.Tensor:
    c = cos(pred, target)
    return torch.clamp(1.0 - c - margin, min=0.0).mean()

optimizer = torch.optim.AdamW(aligner.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ====================== 7) Training (log metrics per epoch; regularization terms are not divided by batch) ======================
history = {"avg_loss": [], "loss_main": [], "loss_prior": [], "loss_coh": []}

for epoch in range(1, EPOCHS + 1):
    aligner.train()
    steps = 0
    sum_main = 0.0
    sum_prior = 0.0
    sum_coh = 0.0
    sum_total = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        if batch is None:
            continue
        flat_idx, seg, lens, targets = batch

        flat_idx = flat_idx.to(DEVICE, non_blocking=True)
        seg = seg.to(DEVICE, non_blocking=True)
        lens = lens.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        preds = aligner(flat_idx, seg, lens)             # [B, d_enc]
        loss_main = cosine_loss(preds, targets)          # Already averaged within the batch
        loss_prior = LAMBDA_PRIOR * aligner.prior_reg()  # Independent of batch, not divided by B
        loss_coh = LAMBDA_COH * aligner.coh_reg(W_neighbors)

        loss = loss_main + loss_prior + loss_coh

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(aligner.parameters(), 1.0)
        optimizer.step()

        steps += 1
        sum_main += loss_main.item()
        sum_prior += loss_prior.item()
        sum_coh += loss_coh.item()
        sum_total += loss.item()

        pbar.set_postfix(avg_loss=f"{(sum_total/steps):.4f}")

    avg_loss = sum_total / steps
    avg_main = sum_main / steps
    avg_prior = sum_prior / steps
    avg_coh = sum_coh / steps

    history["avg_loss"].append(avg_loss)
    history["loss_main"].append(avg_main)
    history["loss_prior"].append(avg_prior)
    history["loss_coh"].append(avg_coh)

    print(f"[Epoch {epoch}] avg_loss={avg_loss:.6f} | loss_main={avg_main:.6f} "
          f"| loss_prior={avg_prior:.6f} | loss_coh={avg_coh:.6f}")

# ====================== 8) Write the learned new token embeddings back to the large model ======================
with torch.no_grad():
    learned_new = aligner.new_embed.weight.detach().to(model.get_input_embeddings().weight.dtype).to(base_device)
    for local_idx, global_id in local_index2global_id.items():
        model.get_input_embeddings().weight[global_id] = learned_new[local_idx]

# If the model's weights are not tied, sync to lm_head (it's automatic when tied)
is_tied = bool(getattr(model.config, "tie_word_embeddings", True))
if not is_tied and getattr(model, "lm_head", None) is not None:
    with torch.no_grad():
        for local_idx, global_id in local_index2global_id.items():
            model.lm_head.weight[global_id] = learned_new[local_idx]

# ====================== 9) Saving and Plotting ======================
os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenizer_new.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print(f"Saved updated model & tokenizer to: {OUTPUT_DIR}")

with open(os.path.join(OUTPUT_DIR, "loss_history.json"), "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

epochs = list(range(1, EPOCHS + 1))
plt.figure(figsize=(8, 5), dpi=150)
plt.plot(epochs, history["avg_loss"], label="avg_loss")
plt.plot(epochs, history["loss_main"], label="loss_main")
plt.plot(epochs, history["loss_prior"], label="loss_prior")
plt.plot(epochs, history["loss_coh"], label="loss_coh")
plt.xlabel("Epoch")
plt.ylabel("Loss (step-avg)")
plt.title("Training Loss Curves (100k samples, regularizers not scaled by batch)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
png_path = os.path.join(OUTPUT_DIR, "loss_curves.png")
plt.tight_layout()
plt.savefig(png_path)
print(f"Saved loss curves to: {png_path}")