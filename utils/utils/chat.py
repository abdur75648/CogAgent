# -*- encoding: utf-8 -*-
'''
@File    :   chat.py
@Time    :   2023/05/08 19:10:08
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import requests
from PIL import Image
from io import BytesIO

import torch
from sat.generation.autoregressive_sampling import filling_sequence, stream_filling_sequence, get_masks_and_position_ids_default
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy
from sat.mpu import get_model_parallel_rank
from sat.helpers import print_rank0

def process_image(image_path, img_processor, cross_img_processor,grounding_img_processor, image):
    if image is None:
        if image_path.startswith("http"):
            response = requests.get(image_path, timeout=10)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
    print("image: ", image)
    if image is not None and isinstance(image, Image.Image):
        print_rank0("Processing image...")
        pil_img = image.convert('RGB')
        img_dict = img_processor(pil_img)
        cross_img_dict = cross_img_processor(pil_img) if cross_img_processor is not None else {}
        gndimg, gnd_ratios = grounding_img_processor.process(pil_img)
        gnd_dict = {'gndimg': gndimg, 'ratios': gnd_ratios}
        ret = (img_dict, pil_img, cross_img_dict, gnd_dict)
    else:
        ret = image
    return ret

def chat(image_path, model, text_processor, img_processor,grounding_img_processor,
        query: str, history: List[Tuple[str, str]] = None, cross_img_processor=None, image: Image = None,
        max_length: int = 4096, top_p=0.95, top_k=5, temperature=0.95, repetition_penalty=1.0,
        invalid_slices=[], no_prompt=False, args=None
        ):
    if image is None:
        assert image_path is not None
    if not history:
        history = []

    if no_prompt:
        query = ''
    prompt = text_processor.history_to_prompt(query, history)

    (torch_image, pil_img, cross_image, gnd_dict) = process_image(image_path, img_processor, cross_img_processor, grounding_img_processor, image)

    if torch_image is not None:
        for k in torch_image:
            if type(torch_image[k]) is torch.Tensor and torch_image[k].dtype is not torch.int and torch_image[k].dtype is not torch.long:
                torch_image[k] = torch_image[k].to(torch.bfloat16 if args.bf16 else torch.float16)
            if type(torch_image[k]) is torch.Tensor:
                torch_image[k] = torch_image[k].to(next(model.parameters()).device)
                
    if cross_image is not None:
        for k in cross_image:
            if type(cross_image[k]) is torch.Tensor and cross_image[k].dtype is not torch.int and cross_image[k].dtype is not torch.long:
                cross_image[k] = cross_image[k].to(torch.bfloat16 if args.bf16 else torch.float16)
            if type(cross_image[k]) is torch.Tensor:
                cross_image[k] = cross_image[k].to(next(model.parameters()).device)
    
    if gnd_dict is not None:
        for k in gnd_dict:
            if type(gnd_dict[k]) is torch.Tensor and gnd_dict[k].dtype is not torch.int and gnd_dict[k].dtype is not torch.long:
                gnd_dict[k] = gnd_dict[k].to(torch.bfloat16 if args.bf16 else torch.float16)
            if type(gnd_dict[k]) is torch.Tensor:
                gnd_dict[k] = gnd_dict[k].to(next(model.parameters()).device)

    print("prompt: ", prompt)
    inputs_dic = text_processor(prompt)
    for k in inputs_dic:
        if type(inputs_dic[k]) is torch.Tensor and inputs_dic[k].dtype is not torch.int and inputs_dic[k].dtype is not torch.long:
            inputs_dic[k] = inputs_dic[k].to(torch.bfloat16 if args.bf16 else torch.float16)
        if type(inputs_dic[k]) is torch.Tensor:
            inputs_dic[k] = inputs_dic[k].to(next(model.parameters()).device)
    input_ids = inputs_dic['input_ids'].to(model.parameters().__next__().device)[0]
    
    if max_length-len(input_ids) <= 1:
        response = "The prompt exceeds the context length limit, please try again."
        return response, history, (torch_image, pil_img)
    
    seq = torch.cat(
        [input_ids, torch.tensor([-1]*(max_length-len(input_ids)), device=input_ids.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[text_processor.tokenizer.eos_token_id],
                            invalid_slices=invalid_slices, repetition_penalty=repetition_penalty)
    # use beam search to get a better result
    # strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[text_processor.tokenizer.eos_token_id],
    #                               num_beams=5, consider_end=True, repetition_penalty=repetition_penalty)
    get_func = text_processor.get_func(input_ids, **inputs_dic) if hasattr(text_processor, 'get_func') else get_masks_and_position_ids_default

    img_inputs = {'vision_'+k: v for k, v in torch_image.items()}
    if cross_image is not None:
        img_inputs = {**img_inputs, **{'cross_'+k:v for k,v in cross_image.items()}}
    if gnd_dict is not None:
        img_inputs = {**img_inputs, **gnd_dict}
    inputs_dic.pop('input_ids')
    inputs = {**img_inputs, **inputs_dic}
    
    # Also set inputs['offset'] as a pytorch tensor([0, 1]) -> Harcoded for now
    inputs['offset'] = torch.tensor([0, 1], device=model.parameters().__next__().device)
    # Set bbox_gt_list as a list (1 single element) of 2-D tensors - Add a dummy tensor for now
    inputs['bboxes_gt_list'] = [torch.tensor([[0, 0, 0, 0]], device=model.parameters().__next__().device)]

    if args.stream_chat:
        raise NotImplementedError # For Grounding, we haven't implemented stream_filling_sequence yet
        filling_stream = stream_filling_sequence(
            model, seq,
            batch_size=1,
            get_masks_and_position_ids=get_func,
            strategy=strategy,
            **inputs
        )
        if get_model_parallel_rank() == 0:
            if 'chinese' in args and not args.chinese:
                print("Model: ", end='')
            else:
                print("模型：", end='')
        offset = len(text_processor.tokenizer.decode(input_ids))
        for tokens, mems in filling_stream:
            torch.cuda.empty_cache()
            tmp_response = text_processor.tokenizer.decode(tokens[0])
            if tmp_response[-1] != "�":
                if get_model_parallel_rank() == 0:
                    tmp_response_offseted = tmp_response[offset:]
                    if hasattr(text_processor, 'process_response'):
                        tmp_response_offseted = text_processor.process_response(tmp_response_offseted)
                    print(tmp_response_offseted, end='', flush=True)
                offset = len(tmp_response)
        if get_model_parallel_rank() == 0:
            print()
        output = strategy.finalize(tokens, mems)[0]

        response = text_processor.tokenizer.decode(output[0])
    else:
        # output = filling_sequence(
        output, bbox_outputs_dict = filling_sequence(
            model, seq,
            batch_size=1,
            get_masks_and_position_ids=get_func,
            strategy=strategy,
            **inputs
        )
        
        # print("output: ", output.shape)
        
        # # )[0] # drop memory
        output = output[0]
        
        print("output: ", output.shape)
        
        # ---------------
        # port from inference_glm.py, more general than chat mode
        # clip -1s and fill back generated things into seq
        if type(output) is not list:
            output_list = output.tolist()
        else:
            output_list = output

        response = text_processor.tokenizer.decode(output_list[0])
    print('original:', response)
    if hasattr(text_processor, 'process_response'):
        response = text_processor.process_response(response)
    # print('processed:', response)
    response = response.split(text_processor.sep)[-1].strip()
    print('final:', response)
    if get_model_parallel_rank() == 0:
        from utils.utils.grounding_parser import parse_response
        parse_response(pil_img, response)
    history = history + [(query, response)]
    return response, history, (torch_image, pil_img, cross_image), bbox_outputs_dict
    # return text_processor,output , history, (torch_image, pil_img, cross_image), bbox_outputs_dict
