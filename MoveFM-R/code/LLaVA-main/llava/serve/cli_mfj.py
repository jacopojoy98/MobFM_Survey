import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import transformers
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

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


# def load_image(image_file):
#     if image_file.startswith('http://') or image_file.startswith('https://'):
#         response = requests.get(image_file)
#         image = Image.open(BytesIO(response.content)).convert('RGB')
#     else:
#         image = Image.open(image_file).convert('RGB')
#     return image


def load_image(image_file,num_id=0):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        # image = Image.open(image_file).convert('RGB')
        image = np.load(image_file)[num_id]

    return image


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    input_ids, targets = [], []
    assistant_tag_len = len([128006,  78191, 128007,271])


    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]
            print('error')

        input_id, target = [], []

        #如果没有system 信息
        input_id += [128000]
        target += [IGNORE_INDEX] * len(input_id)
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

                system_message = "<|start_header_id|>"+role+"<|end_header_id|>"+"\n\n"+content+"<|eot_id|>"
                encode_id = tokenizer(system_message).input_ids[1:]
                input_id += encode_id

                if role in ["user", "system"]:
                    target += [IGNORE_INDEX] * len(encode_id)
                else:
                    target += [IGNORE_INDEX] * assistant_tag_len + encode_id[assistant_tag_len:]
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def data_process_first(tokenizer,image, i,text) -> Dict[str, torch.Tensor]:
    sources = {
    "id": "0",
    "seq": "0,100",
    "conversations": [
        {
            "from": "human",
            "value": "<sequence>\n"
        },
        {
            "from": "gpt",
            "value": "None"
        }
    ]
    }
    sources["conversations"][0]["value"] +=text
    print("input_prompt: ",sources["conversations"][0]["value"])
    if isinstance(i, int):
        sources = [sources]
    assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
    if True:#if 'seq' in self.list_data_dict[i]: #todo
        seq = torch.tensor(image) #todo  函数的输入加入image
        sources = copy.deepcopy([e["conversations"] for e in sources])
    else:
        sources = copy.deepcopy([e["conversations"] for e in sources])
    data_dict = preprocess_llama3(
        sources,
        tokenizer,
        has_image=True) #todo  加入判断，只有第一次才会是true,但是llava 全部都给了，这个看情况，先全部都给看看效果
    if isinstance(i, int):
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])
    # image exist in the data
    is_multimodal = 1 #todo  看有效果没有
    if True:  #'seq' in self.list_data_dict[i]:  #todo
        data_dict['seq'] = seq
    elif is_multimodal:
        data_dict['seq'] = torch.zeros(1, 100, 5)
    
    return data_dict



def data_process_final(instances: Sequence[Dict],tokenizer) -> Dict[str, torch.Tensor]:
    input_ids, labels = tuple([instance[key] for instance in instances]
                              for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX)
    input_ids = input_ids[:, :tokenizer.model_max_length]
    labels = labels[:, :tokenizer.model_max_length]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
    if True: #:'seq' in instances[0]:
        images = [instance['seq'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images
    print("batch:",batch)
    return batch
def main(args):
    # Model
    disable_torch_init()
    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    #---------------------------
    context_len = 4096
    image_processor = None
    output_model_path = "/home/jovyan/workspace/intentLMAllDataset/清华实验目录/mfj_data/llava_try/LLaVA-main_behavior/mfj_save_1227/tot_no_profile_no_mm_1013_5000"
    mm_projector_path = os.path.join(output_model_path, "mm_projector.pth")

    model = LlavaLlamaForCausalLM.from_pretrained(output_model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(output_model_path)

    if True:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    model.get_model().initialize_mm_modules(mm_projector_path)

    #-------------------

    model = model.cuda().to(dtype=torch.bfloat16)
    image_token_index = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    assert image_token_index == 128256, "llama3_image_token_index not match DEFAULT_IMAGE_Index"  # FIXME

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    args.image_file = '/home/jovyan/workspace/intentLMAllDataset/清华实验目录/mfj_data/llava_try/llava_data_0110/seq_data_0110_tot_no_user_profile_train_25wan.npy' 
    image = load_image(args.image_file)
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        # if image is not None:
        #     # first message
        #     if model.config.mm_use_im_start_end:
        #         inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        #     else:
        #         inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        #     image = None
        
        info_dict = data_process_first(tokenizer,image, 0,inp) 
        batch = data_process_final(info_dict,tokenizer)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         input_ids,
        #         images=image_tensor,
        #         image_sizes=[image_size],
        #         do_sample=True if args.temperature > 0 else False,
        #         temperature=args.temperature,
        #         max_new_tokens=args.max_new_tokens,
        #         streamer=streamer,
        #         use_cache=True)

        # batch =  data_module['data_collator'](batch_final)
        input_ids = batch['input_ids'].cuda()
        images = batch['images'].cuda()
        labels = batch['labels'].cuda()
        attention_mask= batch['attention_mask'].cuda()
        print(input_ids)
        print(input_ids.shape)
        # print(labels)
        # input_token_len = input_ids.shape[1]
        output_ids = model.generate(
            input_ids,
            # images=image_tensor.unsqueeze(0).half().cuda(),
            images=images,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            num_beams=3,
            top_k=50,
            attention_mask =attention_mask,
            # no_repeat_ngram_size=3,
            max_new_tokens=400,
            use_cache=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id
            )
        # print(output_ids,"eos_token",tokenizer.eos_token_id)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print('my_output:',outputs)
        # outputs = tokenizer.decode(output_ids[0]).strip()
        # conv.messages[-1][-1] = outputs

        # if args.debug:
        #     print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
