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
    print(tokenizer.pad_token,'tokenizer.pad_token')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(tokenizer.pad_token_id)
    # tokenizer.padding_side = "left"
    # pad_token=model.config.eos_token_id
    #todo
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    #----------------------

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    llava_pretrain_seq_data = np.load(args.image_folder)

    # logits_processor = LogitsProcessorList()
    # logits_processor.append(MinLengthLogitsProcessor(15, eos_token_id=tokenizer.eos_token))
    # logits_processor.append(InfNanRemoveLogitsProcessor())


    for line in tqdm(questions):
        idx = line["id"]
        image_file = line["seq"]
        qs = line["conversations"]
        answer= line['label']
        cur_prompt = qs
        
        if model.config.mm_use_im_start_end:
            pr('123')
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            # print(qs)
            # qs = qs.replace(DEFAULT_IMAGE_TOKEN,'')
            # qs = qs.replace('\n','')
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            # print(qs)
            # pr('12')
        # qs = '1 + 1 ='
        print(qs)
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        print(prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        print(input_ids)
        # image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        # image_tensor = process_images([image], image_processor, model.config)[0]

        now_count = int(image_file.split(",")[0])
        jieceng  = int(image_file.split(",")[1])
        seq = torch.tensor(llava_pretrain_seq_data[now_count,-jieceng:,:]).cuda()

        images = [seq]
        if all(x is not None and x.shape == images[0].shape for x in images):
            # print('123')
            images = torch.stack(images)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                # images=image_tensor.unsqueeze(0).half().cuda(),
                images=images,
                # image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                # do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=3,
                # top_k=args.top_k,
                no_repeat_ngram_size=3,
                max_new_tokens=150,
                use_cache=True
                # logits_processor=logits_processor
                )
        # sampling_params = SamplingParams(temperature=0.6, top_p=0.9,use_beam_search=False,top_k=50,max_tokens=1024)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print("outputs:",outputs,"label:",answer)
        # ans_id = shortuuid.uuid()
        # ans_file.write(json.dumps({"question_id": idx,
        #                            "prompt": cur_prompt,
        #                            "text": outputs,
        #                            "answer_id": ans_id,
        #                            "model_id": model_name,
        #                            "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()


# temperature=0.6,
# top_p=0.9,
# num_beams=3,
# top_k=50,
# # no_repeat_ngram_size=3,
# max_new_tokens=1024,
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
    args = parser.parse_args()

    eval_model(args)
