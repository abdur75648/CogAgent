import os, sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from functools import partial
import torch
import torch.nn.functional as F
import time
import numpy as np
from sat.mpu import get_model_parallel_world_size
from sat.model import AutoModel
from utils.utils import chat, llama2_tokenizer, llama2_text_processor, get_image_processor, parse_response, get_grounding_image_processor
from utils.models import CogAgentModel, CogVLMModel
from utils.utils import TestItemDataset as ItemDataset
from sat.training.deepspeed_training import inference_main

rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

# Greedy decoding
def greedy_decode(logits, tokenizer):
    logits = logits.squeeze(0)
    pred_ids = torch.argmax(logits, dim=-1)
    pred_ids = pred_ids.tolist()
    pred_str = tokenizer.decode(pred_ids)
    return pred_str

# Top-k top-p filtering
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    ):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Source: https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

# Nuclues sampling (using the top_p parameter)
def nucleus_sampling(logits, tokenizer, top_p=0.9):
    logits = logits.squeeze(0)
    logits = top_k_top_p_filtering(logits, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    pred_ids = torch.multinomial(probs, num_samples=1)
    pred_ids = pred_ids.tolist()
    pred_str = tokenizer.decode(pred_ids)
    return pred_str

def data_collator(examples, cross_image_processor=None):
    def to_tensor(value):
        """Converts lists or numpy arrays to tensors."""
        if isinstance(value, list):
            return torch.tensor(value)
        elif isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return value
    
    def concatenate_tensors(attribute, key):
        """Concatenates tensors for a specific attribute and key."""
        if attribute is None:
            return torch.cat([ex[key] for ex in examples if isinstance(ex[key], torch.Tensor)])
        else:
            return torch.cat([ex[attribute][key] for ex in examples if isinstance(ex[attribute][key], torch.Tensor)])

    # Convert all lists and numpy arrays in examples to tensors
    for example in examples:
        for key, value in example.items():
            if isinstance(value, list):
                continue
            example[key] = to_tensor(value)

    # Extract and concatenate attributes from examples
    img_args = {}
    for attribute in ['vision', 'cross']:
        if attribute == 'cross' and cross_image_processor is None:
            continue

        if attribute in examples[-1]:  # Using the last example as reference
            for key in examples[-1][attribute]:
                tensor_key = f"{attribute}_{key}"
                tensors_to_concatenate = [ex[attribute][key] for ex in examples if isinstance(ex[attribute][key], torch.Tensor)]
                if tensors_to_concatenate:
                    img_args[tensor_key] = concatenate_tensors(attribute, key)
                else:
                    img_args[tensor_key] = examples[-1][attribute][key]

    # Remove 'vision' and 'cross' keys from examples
    for example in examples:
        example.pop('vision', None)
        example.pop('cross', None)

    # Create model_args by concatenating tensors and copying other attributes
    model_args = {key: concatenate_tensors(None, key) 
                  if isinstance(examples[-1][key], torch.Tensor) else examples[-1][key] 
                  for key in examples[-1]
                  }
    
    # Merge img_args into model_args
    model_args.update(img_args)
    
    # Add 'offset' key to model_args
    model_args['offset'] = torch.arange(0, model_args['input_ids'].size(0)+1, device=model_args['input_ids'].device)
    
    return model_args

def load_model(args): 
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=0,
        rank=0,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        fp16=args.fp16,
        bf16=args.bf16,
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cpu' if args.quant else 'cuda',
        vg_token_idx = args.vg_token_idx),
        overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {}
    )
    model = model.eval()
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else args.version
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=language_processor_version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
    grounding_image_processor = get_grounding_image_processor(args.gnd_image_pix)

    if args.quant:
        from sat.quantization.kernels import quantize
        quantize(model, args.quant)
        if torch.cuda.is_available():
            model = model.cuda()

    text_processor_infer = llama2_text_processor(tokenizer, args.max_length, model.image_length)

    return model, image_processor, cross_image_processor, text_processor_infer, grounding_image_processor

def create_dataset_function(image_processor, text_processor, cross_image_processor, grounding_image_processor, vg_token, path, args):
    dataset = ItemDataset(image_processor, text_processor, args, path, cross_image_processor=cross_image_processor, grounding_image_processor = grounding_image_processor,vg_token=vg_token)
    return dataset

from sat import mpu
from collections import defaultdict
def broadcast_auto(data_dict):
    type2list = defaultdict(list)
    other = []
    for k in data_dict:
        if type(data_dict[k]) is torch.Tensor:
            type2list[data_dict[k].dtype].append(k)
        else:
            other.append(k)
    new_data = {}
    for k in type2list:
        new_data.update(mpu.broadcast_data(type2list[k], data_dict, k))
    for k in other:
        new_data[k] = data_dict[k]
    return new_data

def get_batch(data_iterator, args, timers):
    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = broadcast_auto(data)
    for k in data_b:
        if type(data_b[k]) is torch.Tensor and data_b[k].dtype is not torch.int32 and data_b[k].dtype is not torch.long:
            if args.fp16:
                data_b[k] = data_b[k].half()
            elif args.bf16:
                data_b[k] = data_b[k].bfloat16()
    return data_b

def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    data_b = get_batch(
        data_iterator, args, timers)
    labels = data_b.pop('labels')
    timers('batch generator').stop()
    
    model_outputs = model(**data_b)
    logits = model_outputs[0][0]
    bbox_outputs_dict = model_outputs[1]
    
    lm_logits = logits.to(torch.float32)
    bbox_outputs = bbox_outputs_dict['bbox_outputs']
    
    return lm_logits, bbox_outputs

def main(args):
    from utils.utils import llama2_tokenizer
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    vg_token = "给"
    args.vg_token_idx = tokenizer.convert_tokens_to_ids(vg_token)
    print("Total number of tokens: ", tokenizer.vocab_size)
    print("Using VG token: ", vg_token, " with index: ", args.vg_token_idx)
    print("\n\nargs:", args)
    assert args.use_lora == True
    
    model, image_processor, cross_image_processor, text_processor_infer, grounding_image_processor = load_model(args)
    # model = None
    # image_processor = None
    # cross_image_processor = None
    # text_processor_infer = None
    # grounding_image_processor = None
    
    
    print("Model Ready For Inference", flush=True)
    print("Model Size: ", sum(p.numel() for p in model.parameters())/1e6, "M", flush=True)
    
    # Create a dataset with given collate_fn and batch_size 1
    dataset_valid = create_dataset_function(image_processor, text_processor_infer, cross_image_processor, grounding_image_processor, vg_token, path="../test_data/apollo_ferret_noscale.json", args=args)
    data_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=1, collate_fn=partial(data_collator, cross_image_processor=cross_image_processor))
    
    # data_iterator = iter(data_loader)
    # from sat.training.utils import Timers
    # timers = Timers()
    # data_b = get_batch(data_iterator, args, timers)
    # print("data_b: ", data_b)

    lm_logits, bbox_outputs = inference_main(args, model=model, forward_step_function=forward_step, data_loader=data_loader)
    
    print("Inference Done", flush=True)
    # print("lm_logits: ", lm_logits)
    # print("bbox_outputs: ", bbox_outputs['pred_boxes'])
    
    output_text_greedy = greedy_decode(lm_logits, tokenizer)
    print("Greedy Decoding: ", output_text_greedy)
    
    output_text_nucleus = nucleus_sampling(lm_logits, tokenizer, top_p=0.5)
    print("Nucleus Sampling: ", output_text_nucleus)
    
    ### NMS -> IMPLEMENT PROPELY ###
    # import torchvision
    # def nms(boxes, scores, iou_threshold=0.5):
    #     indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    #     return indices
    # indices = nms(bbox_outputs['pred_boxes'], bbox_outputs['pred_scores'])
    # print("NMS Indices: ", indices)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_pretrained", type=str, default="../finetune_demo/checkpoints/finetune-cogagent-vqa-03-21-19-37/", help='Path to trained model')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--max_length", type=int, default=1024, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--version", type=str, default="chat_old", choices=['chat', 'vqa', 'chat_old', 'base'], help='version of language process. if there is \"text_processor_version\" in model_config.json, this option will be overwritten')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stream_chat", action="store_true")
    parser.add_argument("--gnd_image_pix", type=int, default=512, help='image pixel size for grounding processor')
    parser.add_argument("--use_lora", action="store_true")
    args = parser.parse_args()
    args = parser.parse_args()
    args.fp16 = True
    args.use_lora = True
    main(args)
