# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import jsonlines

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
import numpy as np
from llava.utils import disable_torch_init
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from transformers import Qwen2ForCausalLM
from torch import autocast

local_rank = None

# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    def __post_init__(self):
        # Set the MASTER_PORT environment variable before initialization
        os.environ['MASTER_PORT'] = '29505'
        super().__post_init__()

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        # if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
        #     if not ignore_status:
        #         logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias" 
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    else:
        return sources

    # for source in sources:
    #     # For pre-training, it means removing the original <image> from <image>, adding an <image> at the beginning, and a newline character '\n'
    #     for sentence in source:
    #         if sentence['value'] is None:
    #             continue
    #         if DEFAULT_IMAGE_TOKEN in sentence['value']:
    #             sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
    #             sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
    #             sentence['value'] = sentence['value'].strip()
    #             if "mmtag" in conversation_lib.default_conversation.version:
    #                 print("mmtag !!!")
    #                 sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Sequence>' + DEFAULT_IMAGE_TOKEN + '</Sequence>')
    #         replace_token = DEFAULT_IMAGE_TOKEN
    #         if data_args.mm_use_im_start_end:
    #             replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    #         sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    # return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

# TODO Not yet finished modifying for llava-next
def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    # image_token_index = -10000,
    # system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}


    # nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    assistant_tag_len = len([128006,  78191, 128007,271])


    for i, source in enumerate(sources):

        # if plain_flag:
        #     # print('111111111111111')
        #     assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        #     source[0]['value'] = DEFAULT_IMAGE_TOKEN

        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []


        # If there is no system message
        input_id += [128000]


        target += [IGNORE_INDEX] * len(input_id)
        
        # print('system',tokenizer.decode(input_id, skip_special_tokens=False),'decoded_text')


        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]

            if content is None:
                encode_id = [128006,  78191, 128007,271]
                input_id +=encode_id
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                # print(conv,'conv')
                # First is bos token we don't need here
                system_message = "<|start_header_id|>"+role+"<|end_header_id|>"+"\n\n"+content+"<|eot_id|>"
                # print(system_message,'++++')
                encode_id = tokenizer(system_message).input_ids[1:]
                # encode_id = tokenizer.apply_chat_template(conv)[1:]
                input_id += encode_id
                # print(encode_id,'encode_id')
                if role in ["user", "system"]:
                    target += [IGNORE_INDEX] * len(encode_id)
                    # decoded_text = tokenizer.decode(encode_id, skip_special_tokens=False)
                    # print('user',decoded_text,'decoded_text')
                else:
                    target += [IGNORE_INDEX] * assistant_tag_len + encode_id[assistant_tag_len:]
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)
    # print(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)


    # pr(123)
    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )






def preprocess_qwen2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    # image_token_index = -10000,
    # system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
    system_message: str = "none",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Apply prompt templates
    input_ids, targets = [], []
    # assistant_tag_len = len([128006,  78191, 128007,271]) # This is for llama
    assistant_tag_len = len([151644, 77091, 198])


    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # print('qwen2-begin---------------------------------------------')
    for i, source in enumerate(sources):

        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # print(tokenizer.convert_tokens_to_ids("\n\n"),'271')

        # New version, use apply chat template
        # Build system message for each sentence

        if system_message != 'none':
            # Has system message
            # system_message = "<|start_header_id|>system<|end_header_id|>"+"\n\n"+system_message+"<|eot_id|>"
            # input_id += tokenizer(system_message).input_ids

            input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
            target += [IGNORE_INDEX] * len(input_id)


        # def build_qwen_prompt(conversation,system_info=None):
        #     prompt = "<|im_start|>system\n{}<|im_end|>\n".format(system_info)
        #     for msg in conversation[:-1]:
        #         if msg['from']=="gpt":
        #             # msg['from']=="assistant"
        #             prompt += "<|im_start|>assistant\n{}<|im_end|>\n".format(msg['value'])
        #         if msg['from']=="human":
        #             # msg['from']=="user"
        #             prompt += "<|im_start|>user\n{}<|im_end|>\n".format(msg['value'])
        #         # prompt += "<|im_start|>{}\n{}<|im_end|>\n".format(msg['from'],msg['value'])
        #     prompt += "<|im_start|>assistant\n"
        #     return prompt,conversation[3]["value"]


        
        # print('system',tokenizer.decode(input_id, skip_special_tokens=False),'decoded_text')
        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # encode_id = tokenizer.apply_chat_template(conv)
            # print(encode_id)
            # pr(123)
            if content is None:
                # TODO  <|im_start|>assistant\n
                encode_id = [151644, 77091, 198]
                input_id +=encode_id
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                encode_id = tokenizer.apply_chat_template(conv)
                # TODO qwen does not have an eos token, so there is no need to remove the initial eos token
                # encode_id = tokenizer.apply_chat_template(conv).input_ids[1:] 
                input_id += encode_id
                if role in ["user", "system"]:
                    target += [IGNORE_INDEX] * len(encode_id)
                else:
                    # target += encode_id
                    target += [IGNORE_INDEX] * assistant_tag_len + encode_id[assistant_tag_len:]

                # system_message = "<|start_header_id|>"+role+"<|end_header_id|>"+"\n\n"+content+"<|eot_id|>"
                # # print(system_message,'++++')
                # encode_id = tokenizer(system_message).input_ids[1:]
                # encode_id = tokenizer.apply_chat_template(conv)[1:]
                # input_id += encode_id
                # print(encode_id,'encode_id')
                # if role in ["user", "system"]:
                #     target += [IGNORE_INDEX] * len(encode_id)
                # else:
                #     target += [IGNORE_INDEX] * assistant_tag_len + encode_id[assistant_tag_len:]
            # pr(123)
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)
    # print(input_ids,'input_ids')
    # print('targets:',targets)
    # print('sourece',sources)
    # pr(123)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


# try More efficient in understanding, but not actually in efficiency
def preprocess_llama3_try(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    input_ids, targets = [], []
    # assistant_tag_len = len([128006,  78191, 128007,271])
    assistant_tag_len = 4

    for source in sources:  
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']

        input_id, target = [], []

        system_message = "<|start_header_id|>"+"user"+"<|end_header_id|>"+"\n\n"+source[0]['value']+"<|eot_id|>"
        encode_id = tokenizer(system_message).input_ids

        input_id += encode_id
        target += [IGNORE_INDEX] * len(input_id)

        if source[1]['value'] is None:
            encode_id = [128006,  78191, 128007,271]
            input_id +=encode_id
            target += [IGNORE_INDEX] * len(encode_id)
        else:
            system_message = "<|start_header_id|>"+"assistant"+"<|end_header_id|>"+"\n\n"+source[1]['value']+"<|eot_id|>"
            encode_id = tokenizer(system_message).input_ids[1:]

            input_id += encode_id
            target += [IGNORE_INDEX] * assistant_tag_len + encode_id[assistant_tag_len:]
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)

    # print(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


"qwen2"
def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    system_message: "none",
) -> Dict:
    # add end signal and concatenate together
    # conversations = []

    input_ids, targets = [], []
    # assistant_tag_len = len([128006,  78191, 128007,271])
    assistant_tag_len = len([151644, 77091, 198])
    # IMAGE_TOKEN_INDEX
    user_input_ids = [151644, 872, 198]+ [IMAGE_TOKEN_INDEX]+ [715, 151645, 198]


    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # system_info = [151644, 8948, 198, 2610, 525, 264, 7785, 6371, 7709, 17847, 369, 3847, 13, 1446, 646, 3535, 35595, 1995, 323, 1196, 25785, 504, 279, 39088, 8500, 315, 862, 13656, 27099, 11, 323, 27079, 2736, 9079, 1741, 438, 7709, 8660, 11, 19639, 11, 323, 9471, 13, 71984, 1995, 5646, 25, 10159, 315, 279, 2003, 320, 68, 1302, 2572, 7014, 11, 7589, 701, 1462, 315, 1899, 320, 258, 364, 23180, 25, 8035, 6, 3561, 11, 384, 1302, 2572, 220, 17, 18, 25, 19, 20, 701, 4707, 3034, 320, 68, 1302, 2572, 9866, 220, 15, 701, 1556, 943, 320, 68, 1302, 2572, 70, 6469, 11, 6154, 4292, 151645, 198]



    for source in sources:  
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        # source[0]['value'] = DEFAULT_IMAGE_TOKEN
        # print(source[1]['value'],'label')
        # conversation = source[0]['value'] + conversation_lib.default_conversation.sep

        

        input_id, target = [], []

        if system_message != 'none':
            input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
            target += [IGNORE_INDEX] * len(input_id)

        input_id += user_input_ids
        target += [IGNORE_INDEX] * len(user_input_ids)

        # system_message = "<|start_header_id|>"+"assistant"+"<|end_header_id|>"+"\n\n"+source[1]['value']+"<|eot_id|>"
        # encode_id = tokenizer(system_message).input_ids
        if source[1]['value'] is not None:
            encode_id = tokenizer.apply_chat_template([{"role" : "assistant", "content" : source[1]['value']}])

            input_id += encode_id
            target += [IGNORE_INDEX] * assistant_tag_len + encode_id[assistant_tag_len:]
        else:

            encode_id = [151644, 77091, 198]
            input_id +=encode_id
            target += [IGNORE_INDEX] * len(encode_id)


        input_ids.append(input_id)
        targets.append(target)
    # print(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:


    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # roles = {"human": USER, "gpt": ASSISTANT}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )




def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    system_message: str = "none"
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # print(conversation_lib.default_conversation.version)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer,system_message=system_message)
        # return preprocess_llama3(sources, tokenizer, has_image=has_image,plain_flag=1)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen2":
        # print('choose qwen2')
        return preprocess_qwen2(sources, tokenizer, has_image=has_image,system_message=system_message)

    # pr(123)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 eval_flag = 0,
                 mask_flag = 0,
                 eval_dataset = 0,
                 predict_flag = 0,
                 chat_stage = 2):
        super(LazySupervisedDataset, self).__init__()

        list_data_dict = json.load(open(data_path, "r"))


        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.eval_flag = eval_flag
        self.mask_flag = mask_flag
        self.predict_flag = predict_flag
        self.chat_stage = chat_stage
        self.tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        

        if self.chat_stage is not None:
            list_data_dict_final = []
            for item in self.list_data_dict:
                item['conversations'] = item['conversations'][0:self.chat_stage*2]
                list_data_dict_final.append(item)
            self.list_data_dict = list_data_dict_final
            print(len(self.list_data_dict))
        # 0318_task3_generate_trun1_2600

        image_token_index = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN) 
        # print(image_token_index,"llama3_image_token_index not match DEFAULT_IMAGE_Index") #151665
        # assert image_token_index == 128256, "llama3_image_token_index not match DEFAULT_IMAGE_Index"  # FIXME
        

        self.llava_pretrain_seq_data = np.load(self.data_args.image_folder) 



        # self.llava_pretrain_seq_data = np.load(self.data_args.image_folder)
        print('self.llava_pretrain_text_data.length:',len(self.list_data_dict))
        print('self.llava_pretrain_seq_data.shape:',self.llava_pretrain_seq_data.shape)

        
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # img_tokens = 128 if 'image' in sample else 0
            #notice 'image'!= "<image>"
            img_tokens = int(sample['seq'].split(",")[1]) if  'seq' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # cur_len = cur_len if 'image' in sample else -cur_len
            cur_len = cur_len if 'seq' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        if self.eval_flag:
            sources['conversations'][-1]["value"] = None


        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if 'system' in sources[0]:
            system_message = sources[0]['system']
        else:
            system_message =None

        if 'seq' in sources[0]:
            

            image_file = self.list_data_dict[i]['seq']
            now_count = int(self.list_data_dict[i]['id'])
            layer_num  = int(image_file.split(",")[1])

            # now_count = 0

            seq = torch.tensor(self.llava_pretrain_seq_data[now_count,-layer_num:,:])


            if self.mask_flag :
                mask_pos = torch.tensor(self.llava_pretrain_seq_data_mak_pos[now_count])

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])


        # Add system message
        if system_message is not None:
            data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('seq' in self.list_data_dict[i]),
            system_message = system_message
            )
        else:
            data_dict = preprocess(
                sources,
                self.tokenizer,
                has_image=('seq' in self.list_data_dict[i]))

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'seq' in self.list_data_dict[i]:
            data_dict['seq'] = seq
            # print("type(seq)",type(seq))
            if  self.mask_flag:
                # mask_pos = self.llava_pretrain_seq_data_mak_pos[now_count]
                data_dict['mask_pos'] = mask_pos
                # print("type(mask_pos)",type(mask_pos))

        
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['seq'] = torch.zeros(1, 3, 5)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'seq' in instances[0]:
            images = [instance['seq'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

            if 'mask_pos' in instances[0]:
                mask_pos = [instance['mask_pos'] for instance in instances]
                if all(x is not None and x.shape == mask_pos[0].shape for x in mask_pos):
                    batch['mask_pos'] = torch.stack(mask_pos)
                else:
                    batch['mask_pos'] = mask_pos
        # print("batch:",batch)
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # TODO
    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                eval_dataset=1)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

def make_supervised_data_module_eval(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                eval_flag=1)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=eval_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

def make_supervised_data_module_predict(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                predict_flag =1)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

# Second-stage
def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # With pre-training-------------------------------------------------------------
    bnb_model_from_pretrained_args = {}

    print(model_args.model_name_or_path)
    if model_args.vision_tower is not None:

        # true  load -model--------------------------------------------------
        if "qwen" in model_args.model_name_or_path.lower():
            print('loading qwen2')
            model = LlavaQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        else:
            print('loading llama3')
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            # if training_args.local_rank == 0:
            #     print('ini_model',model)
    elif "qwen" in model_args.model_name_or_path.lower():
        # TODO
        print('loading qwen2')
        model = transformers.Qwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    else:
        print('loading llama3')
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            # TODO see comments
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        # model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    # true
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False, # ini finished modification
            # use_fast=True,
        )



    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    # true
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

        # TODO decide whether to add
        if "llama-3" in model_args.model_name_or_path.lower():
            tokenizer.pad_token="padding"

        if "qwen" in model_args.model_name_or_path.lower():
            # print(tokenizer.pad_token)
            # pr(123) 9571
            tokenizer.pad_token="padding"

    # done modify by reference
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        # data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        # print(training_args.tune_mm_mlp_adapter,model_args.tune_mm_mlp_adapter)
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            # print('isisisi')
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        # TODO take a look at this, no need to worry about it for now, the parameters involved are all false, which means this function is not called
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)





    from peft import PeftModel,PeftConfig
    # Merge lora parameters
    
    lora_model_path = 'your_finetune_model_path'
    
    model = PeftModel.from_pretrained(model, lora_model_path)
    # model =model.to(torch.bfloat16)

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules_reload(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()

        print(vision_tower.vision_tower.lm_head.weight.dtype,'vision_tower.dtype---after')

    # model = model.merge_and_unload()  # Key step: merge and unload LoRA layers

    mm_projector_path = os.path.join(lora_model_path, "mm_projector.pth")


    model.get_model().initialize_mm_modules(mm_projector_path)


    

    model.get_model().mm_projector.to(dtype=model.dtype)



    model.requires_grad_(False)
    for p in model.get_vision_tower().parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Merge lora parameters---end------------------------


    # Always keep the vision-tower data type as fp32, do not change
    import types as _types
    if True:
        print("Always keep the vision-tower data type as fp32, do not change")
        def install_fp32_guard(module):
            """
            Locks the `module` (e.g., vision_tower) to FP32:
            - Intercept Module._apply: any attempt to change a tensor to bf16/fp16 is changed back to float32 (device changes are allowed)
            - Intercept Module.to(dtype=...): ignore dtype, only allow device changes
            - Intercept Module.half()/bfloat16(): becomes a no-op
            - Finally, unify to .to(torch.float32) once
            Can be called repeatedly (idempotent), wrapped only once.
            """
            

            # ---- Intercept _apply (most critical: DS's module.bfloat16() will go here) ----
            if not hasattr(module, "_orig_apply_guard"):
                module._orig_apply_guard = module._apply

                def _apply_guard(self, fn, *args, **kwargs):
                    def fn_guard(t: torch.Tensor):
                        t2 = fn(t)
                        # If the target is a floating-point tensor and is converted to half-precision, force it back to float32
                        if t2.is_floating_point() and t2.dtype in (torch.bfloat16, torch.float16):
                            return t2.float()
                        return t2
                    return self._orig_apply_guard(fn_guard, *args, **kwargs)

                module._apply = _types.MethodType(_apply_guard, module)

            # ---- Intercept to(): mask dtype, only allow device changes ----
            if not hasattr(module, "_orig_to_guard"):
                module._orig_to_guard = module.to

                def _to_guard(self, *args, **kwargs):
                    new_args = list(args)
                    # Positional arguments may have directly given a dtype
                    if new_args and isinstance(new_args[0], torch.dtype):
                        new_args[0] = torch.float32
                    # Keyword arguments may also contain a dtype
                    if "dtype" in kwargs and kwargs["dtype"] is not None:
                        kwargs["dtype"] = torch.float32
                    return self._orig_to_guard(*new_args, **kwargs)

                module.to = _types.MethodType(_to_guard, module)

            # ---- Intercept half()/bfloat16(): set to no-op (return self) ----
            if not hasattr(module, "_orig_half_guard"):
                module._orig_half_guard = module.half
                def _half_guard(self, *a, **k): return self
                module.half = _types.MethodType(_half_guard, module)

            if not hasattr(module, "_orig_bf16_guard"):
                module._orig_bf16_guard = module.bfloat16
                def _bf16_guard(self, *a, **k): return self
                module.bfloat16 = _types.MethodType(_bf16_guard, module)

            # ---- Ensure the current type is FP32 ----
            module.to(torch.float32)
            return module

        install_fp32_guard(vision_tower) 

        if hasattr(model, "encode_images"):
            _old = model.encode_images
            def _encode_images_fp32(self, *args, **kwargs):
                kwargs.pop("model_dtype", None)        # Avoid using this internally to change weight dtype
                with autocast('cuda', enabled=False):  # Visual computation is FP32 throughout
                    feats = _old(*args, **kwargs)
                return feats.to(dtype=self.model.dtype)  # Only convert output activations to BF16, do not change weights
            model.encode_images = _encode_images_fp32.__get__(model, type(model))


    # Always keep the vision-tower data type as fp32, do not change---------------end0---------------------------------------



    # print(model)
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]


    print('begin eval-----------------------------------------------------------')



    # Prefix constraint for token generation

    from typing import Dict, List, Tuple, Optional, Set
    from functools import lru_cache
    from transformers import LogitsProcessor, LogitsProcessorList
    # ===== 1) Directly construct "token id sequences" and "A/B/C/D class sets" from a JSON dictionary =====
    def load_token_id_seqs_and_classes(json_path: str, tokenizer,tot_nums=None):
        """
        json is like:
        {
          "0": ["<a_246>", "<b_156>", "<c_30>", "<d_448>"],
          "1": ["<a_275>", "<b_162>", "<c_421>", "<d_185>"],
          ...
        }
        Returns:
          - token_id_seqs: List[List[int]]  Each legal sequence is exactly 4 token ids
          - class_sets: [set(A_ids), set(B_ids), set(C_ids), set(D_ids)]
        """
        with open(json_path, "r", encoding="utf-8") as f:
            legal = json.load(f)  # dict[str, list[str]]

        if tot_nums is not None:
            count = 0
            legal_final = {}
            for key,value in legal.items():
                legal_final[key] = value
                count +=1
                if count>=tot_nums:
                    break
            legal = legal_final
        
        print('tot_legal_sequence_number: ', len(legal))

        token_id_seqs: List[List[int]] = []
        A, B, C, D = set(), set(), set(), set()

        for v in legal.values():
            assert len(v) == 4, f"Each legal sequence must be 4 segments, but got {v}"
            ids = []
            for pos, tok in enumerate(v):
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is None or tid < 0:
                    raise ValueError(f"token {tok} is not in the vocabulary, please confirm that you have added tokens and resized.")
                ids.append(int(tid))
                if   pos == 0: A.add(tid)
                elif pos == 1: B.add(tid)
                elif pos == 2: C.add(tid)
                else:          D.add(tid)
            token_id_seqs.append(ids)

        class_sets = [A, B, C, D]
        return token_id_seqs, class_sets


    # ===== 2) Trie (based on token id) =====
    class _TrieNode:
        __slots__ = ("children", "terminal")
        def __init__(self):
            self.children: Dict[int, "_TrieNode"] = {}
            self.terminal: bool = False

    class TokenTrie:
        def __init__(self):
            self.root = _TrieNode()

        def insert(self, seq: List[int]) -> None:
            node = self.root
            for tid in seq:
                node = node.children.setdefault(tid, _TrieNode())
            node.terminal = True

        def allowed_next(self, prefix: List[int]) -> Tuple[bool, Set[int]]:
            node = self.root
            for tid in prefix:
                if tid not in node.children:
                    return False, set()
                node = node.children[tid]
            if node.terminal:
                return True, set()
            return False, set(node.children.keys())


    def build_trie_from_token_id_seqs(token_id_seqs: List[List[int]]) -> TokenTrie:
        trie = TokenTrie()
        for ids in token_id_seqs:
            assert len(ids) == 4, "Each legal sequence must be 4 token ids"
            trie.insert(ids)
        return trie


    # ===== 3) Custom LogitsProcessor (with robust suffix & positional constraints) =====
    class TrieConstrainedLogitsProcessor(LogitsProcessor):
        def __init__(
            self,
            tokenizer,
            trie: TokenTrie,
            class_sets: List[Set[int]],           # [A_ids, B_ids, C_ids, D_ids]
            eos_fallback: Optional[int] = None,
            prompt_lengths: Optional[List[int]] = None,
            max_slots: int = 4                    # Only allow generating 4 slots (a,b,c,d)
        ):
            super().__init__()
            if eos_fallback is None:
                eos_fallback = (
                    tokenizer.eos_token_id
                    if tokenizer.eos_token_id is not None
                    else (tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.pad_token_id)
                )
            if eos_fallback is None:
                raise ValueError("Unable to determine eos_fallback, please manually pass in a usable token id.")

            self.trie = trie
            self.class_sets = [set(s) for s in class_sets]
            self.eos_fallback = int(eos_fallback)
            self.prompt_lengths = prompt_lengths  # None => Treat uniformly as 0
            self.max_slots = max_slots

            self._cache = lru_cache(maxsize=100_000)(self._allowed_for_suffix)

        def _allowed_for_suffix(self, suffix_tuple: Tuple[int, ...]) -> Tuple[bool, Tuple[int, ...]]:
            terminal, allowed = self.trie.allowed_next(list(suffix_tuple))
            return terminal, tuple(sorted(allowed))

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            device = scores.device
            vocab_size = scores.size(-1)
            B = input_ids.size(0)

            for b in range(B):
                cur_len = input_ids.size(1)

                # -- Robust suffix calculation -- #
                if self.prompt_lengths is None:
                    pl = 0
                else:
                    pl = self.prompt_lengths[b % len(self.prompt_lengths)]
                    # Some wrappers only pass in "generated", in which case pl might be > cur_len
                    if pl > cur_len:
                        pl = 0

                suffix = input_ids[b, pl:].tolist()

                # Slot: The current position to generate is the len(suffix)-th position (0-based)
                slot_idx = len(suffix)

                # Force end if it exceeds 4 segments
                if slot_idx >= self.max_slots:
                    allowed_ids = [self.eos_fallback]
                else:
                    terminal, allowed = self._cache(tuple(suffix))

                    # First constrain with Trie, then intersect with "positional class sets"
                    allowed_ids = [t for t in allowed if 0 <= t < vocab_size]

                    # Intersect with positional class sets (to ensure a->b->c->d order)
                    # allowed_ids = list(set(allowed_ids) & self.class_sets[slot_idx])
                    allowed_ids = list(set(allowed_ids))

                    # No path to go or already at the end -> can only end
                    if terminal or len(allowed_ids) == 0:
                        allowed_ids = [self.eos_fallback]

                # Mask logits
                mask = torch.full((vocab_size,), float("-inf"), dtype=scores.dtype, device=device)
                idx = torch.tensor(allowed_ids, device=device, dtype=torch.long)
                mask.scatter_(0, idx, 0.0)
                scores[b] = scores[b] + mask

            return scores






    # import math
    # eval---------------------------------
    import math
    
 


    "todo: edit your file"
    # "all_task-no_sequence"
    data_args.image_folder = 'data/stage2/so_final_all_task_data_test.npy'  # Path to sequence data
    data_args.data_path= 'data/stage1/s2_final_test_dat_list_id_and_loc_4_city_100wan_new_with_share_gpt.json' # Path to plain text data


    # TODO
    # Load test set
    data_module = make_supervised_data_module_eval(tokenizer=tokenizer,
                                              data_args=data_args)

    train_dataset = data_module['train_dataset']
    print(len(train_dataset))
    # model = trainer.get_my_model()

    # trainer = LLaVATrainer(model=model,
    #                 tokenizer=tokenizer,
    #                 args=training_args,
    #                 **data_module)
    print("Is the embedding weight sharded:", hasattr(model.get_model().embed_tokens.weight, 'ds_id'))


    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    import time


    model = model.cuda().to(dtype=torch.bfloat16)

    # print(model.get_model().mm_projector.dtype,model.dtype)

    # model = model.cuda()
    batch_size = 1
    train_dataset = train_dataset


    for name, param in model.get_model().mm_projector.named_parameters():
        if 'weight' in name:
            print(f"Layer: {name}, dtype: {param.dtype}")
            break

    print(model.dtype,model.get_vision_tower().vision_tower.lm_head.weight.dtype)
    # return
    model.model_max_length = 12500

    current_device = torch.cuda.current_device()
    # Get the ID of the currently used graphics card
    gpu_count = int(torch.cuda.device_count())
    print(gpu_count,current_device,'-----')
    # import time
    # time.sleep(30000)
    tot_numnber = int(len(train_dataset)/batch_size)

    init_number = 0

    print('init_number:',init_number)
    num_start = tot_numnber/gpu_count* current_device + init_number
    num_end = tot_numnber/gpu_count* (current_device+1) + init_number

    if current_device +1 ==gpu_count:
        num_end = tot_numnber + init_number
    print(num_start,num_end,current_device,'num_start,num_end,current_device')


    this_device_init_num = num_start  # Round up
    print(current_device,'current_device++++++++++++++++++++',this_device_init_num)
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from tqdm import tqdm

    list_data_dict = json.load(open(data_args.data_path, "r"))

    results = []
    count_id = 0
    # pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token="padding"
    tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)



    batch_size = 1  # Set batch size to 1
    with torch.inference_mode():
        with tqdm(total=int(num_end-num_start), desc='test', leave=False) as tq:
            # Change the loop step to batch_size
            for j in range(int(num_start), int(num_end), batch_size):
                # Get the index range of the current batch
                current_batch_indices = range(j, min(j+batch_size, int(num_end)))

                # Build batch data
                batch_final = [train_dataset[i] for i in current_batch_indices]

                # Skip empty batches
                if not batch_final:
                    continue

                batch = data_module['data_collator'](batch_final)
                input_ids = batch['input_ids'].cuda()
                if count_id<1:
                    print(input_ids.shape)
                images = batch['images'].cuda()
                labels = batch['labels'].cuda()
                attention_mask = batch['attention_mask'].cuda()

                # Generation task parameters
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True,
                    temperature=1.4,
                    top_p=0.8,
                    top_k=20,
                    attention_mask=attention_mask,
                    max_new_tokens=3000,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # Prediction task parameters
                # output_ids = model.generate(
                #     input_ids,
                #     # images=image_tensor.unsqueeze(0).half().cuda(),
                #     images=images,
                #     do_sample=True,
                #     temperature=0.7,
                #     top_p=0.8,
                #     num_beams=3,
                #     top_k=20,
                #     attention_mask =attention_mask,
                #     max_new_tokens=2000,
                #     use_cache=True,
                #     pad_token_id=tokenizer.pad_token_id,
                #     repetition_penalty = 1.05,
                # )

     
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                # Iterate through and process each sample result
                for idx, output in enumerate(outputs):
                    actual_index = j + idx
                    if actual_index >= len(train_dataset):
                        print('break')
                        continue

                    sorce = list_data_dict[actual_index]
                    label = f"{sorce['conversations'][-1]['value']}"

                    # Keep the original logic for debugging output
                    if count_id < 1:
                        print(f"outputs[{idx}]:", output)
                        print("label:", label)
                        count_id += 1

                    results.append({
                        "prompt": str(sorce['seq'].split(',')[0]),
                        "label": label,
                        "predict": output
                    })

                    tq.update(1)



    print('final')
 
    version = "only_all_teak_no_sequence_generate_all_city"

    result_save_path = '/eval_result_0425/' + version
    if not os.path.exists(result_save_path):
        # If the directory does not exist, create it
        os.makedirs(result_save_path, exist_ok=True)
    # Store the results in JSONL format
    if  gpu_count>1:
        with open(result_save_path + '/eval'+str(current_device)+"split"+".jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

    
    else:

        with open(result_save_path + '/eval'+str(current_device)+".jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    print('Commented out loading of mm module---------------------------------------------------------')

    seed =42#42
    set_all_seeds(seed)
    print("setting seed=",seed)

    train()