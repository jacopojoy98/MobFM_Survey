import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

# from PIL import Image
import math
import numpy as np
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.logits_process import InfNanRemoveLogitsProcessor, MinLengthLogitsProcessor

import warnings

# 过滤特定的警告信息
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # model = model.to(dtype=torch.bfloat16)
    # tokenizer.pad_token='<|eot_id|>'
    data_module = make_supervised_data_module_eval(tokenizer=tokenizer,
                                              data_args=data_args)

    train_dataset = data_module['train_dataset']
    print(len(train_dataset))
    # model = trainer.get_my_model()

    model_path ='/home/jovyan/workspace/intentLMAllDataset/清华实验目录/mfj_data/llava_try/LLaVA-main_behavior/merge_lora_models/llava-Llama-3.1-8b-Instruct_3jie_batch32_3epcoh_20wan_all_llama3'

    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer1, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]

    if "llama-3" in model_args.model_name_or_path.lower():
        tokenizer.pad_token='<|eot_id|>'

    model = model.cuda().to(dtype=torch.bfloat16)
    batch_size = 1
    train_dataset = train_dataset

    current_device = torch.cuda.current_device()
    
    this_device_init_num = math.ceil(len(train_dataset)/batch_size/8)*current_device #向上取整
    print(current_device,'current_device++++++++++++++++++++',this_device_init_num)
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from tqdm import tqdm

    list_data_dict = json.load(open(data_args.data_path, "r"))

    results = []
    pad_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
    # temperature=0.6, top_p=0.9,use_beam_search=False,top_k=50,max_tokens=1024
        with tqdm(total = int(len(train_dataset)/batch_size), desc = 'test',leave=False) as tq:
        # with tqdm(total = math.ceil(len(train_dataset)/batch_size/8), desc = 'test',leave=False) as tq:
            # for i, batch in enumerate(train_dataset):
            # for j in range(10):
            # for j in range(20):
            for j in range(math.ceil(len(train_dataset)/batch_size)):
            # for j in range(math.ceil(len(train_dataset)/batch_size/8)):
                # batch = dict(
                #     input_ids=input_ids,
                #     labels=labels,
                #     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                # )
                # print(batch)
                # print(batch.keys())
                # if this_device_init_num+j >= len(train_dataset):
                #     tq.update(1)
                #     continue
                # batch_final = [train_dataset[i] for i in range(this_device_init_num+j,this_device_init_num+j+batch_size) ]
                
                batch_final = [train_dataset[i] for i in range(j,j+batch_size) ]

                batch =  data_module['data_collator'](batch_final)
                input_ids = batch['input_ids'].cuda()
                images = batch['images'].cuda()
                labels = batch['labels'].cuda()
                attention_mask= batch['attention_mask'].cuda()
                # print(input_ids)
                # print(labels)
                input_token_len = input_ids.shape[1]
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
                    max_new_tokens=10,
                    use_cache=True,
                    pad_token_id=pad_token_id
                    )
                # input_token_len = input_ids.shape[1]
                # print(input_ids.shape,'input_ids.shape')
                # print(output_ids.shape,'output_ids.shape')        
                # print()
                # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                # if n_diff_input_output > 0:
                #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                # labels = tokenizer.batch_decode(labels, skip_special_tokens=True)[0]
                print(outputs)
                # print(labels,'labels')
                tq.update(1)

                # sorce = list_data_dict[this_device_init_num+j]
                sorce = list_data_dict[j]
                label = sorce['conversations'][1]["value"] + ','+str(sorce['seq'].split(",")[0])
                # if current_device == 0:
                #     print(outputs.split('\n')[0:3],'outputs----------',label,'labels')
                results.append({
                    "prompt": str(sorce['seq'].split(",")[0]),
                    "label":label,
                    "predict": outputs
                    })
    print('final')
    version = 'train_samples_20wan_3epcoh_all_llama3_chat'
    result_save_path = '/home/jovyan/workspace/intentLMAllDataset/清华实验目录/mfj_data/llava_try/LLaVA-main_behavior/eval_result/' + version
    if not os.path.exists(result_save_path):
        # 如果目录不存在，则创建目录
        os.makedirs(result_save_path, exist_ok=True)
    # 将结果存储为JSONL格式

    with open(result_save_path + '/eval'+str(current_device)+".jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="plain")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--tokenizer_path", type=str, default="plain")
    parser.add_argument("--vesion", type=str, default="plain")
    args = parser.parse_args()

    eval_model(args)
