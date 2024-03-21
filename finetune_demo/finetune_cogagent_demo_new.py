import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import json
import torch
import argparse
from functools import partial
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from torch.nn import CrossEntropyLoss
from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from sat.helpers import print_rank0
from utils.models import FineTuneTrainCogAgentModelNew as FineTuneTrainCogAgentModel
from utils.utils import llama2_text_processor, llama2_text_processor_inference, get_image_processor, get_grounding_image_processor

def disable_untrainable_params(self):
    total_trainable = 0
    # enable = ['vit']
    enable = ["encoder", "cross_attention", "linear_proj", 'mlp.vision', 'rotary.vision', 'eoi', 'boi', 'vit', 'grounding_fc']
    if self.args.use_ptuning:
        enable.extend(['ptuning'])
    if self.args.use_lora or self.args.use_qlora:
        enable.extend(['matrix_A', 'matrix_B'])
    for n, p in self.named_parameters():
        flag = False
        for e in enable:
            if type(e) is tuple:
                if e[0].lower() in n.lower() and e[1].lower() in n.lower() and 55 > int(n[:n.find('.mlp')].split('.')[-1]) > 45:
                    flag = True
                    break
            else:
                if e.lower() in n.lower():
                    flag = True
                    break
        if not flag:
            p.requires_grad_(False)
        else:
            total_trainable += p.numel()
            if 'encoder' in n or 'vit' in n:
                p.lr_scale = 0.1
            print_rank0(n)
    
    print_rank0("***** Total trainable parameters: "+str(total_trainable)+" *****")

FineTuneTrainCogAgentModel.disable_untrainable_params = disable_untrainable_params

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
    
    logits, bbox_outputs_dict = model(**data_b)
    
    lm_logits = logits.to(torch.float32)
    # Shift so that tokens < n predict n
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = lm_logits[..., -1-shift_labels.size(-1):-1, :].contiguous()
    # print("shift_labels: ", shift_labels.shape)
    # print("Unqiue values in shift_labels: ", torch.unique(shift_labels))
    # print("shift_logits: ", shift_logits.shape)
    # print("Unqiue values in shift_logits: ", torch.unique(shift_logits))
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    llm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    llm_loss = llm_loss.to(torch.float32)
    
    # llm_loss = torch.tensor(0, dtype=torch.float32, device=shift_logits.device)
    
    bbox_outputs = bbox_outputs_dict['bbox_outputs']
    gt_ids = bbox_outputs_dict['gt_ids']
    target = bbox_outputs_dict['target']
    initial_pred_embeddings = bbox_outputs_dict['initial_pred_embeddings']
    gnd_loss_dict = model.get_mixin('grounding').criterion_grounding(bbox_outputs, target, initial_pred_embeddings[gt_ids].unsqueeze(1))
    weight_dict = model.get_mixin('grounding').criterion_grounding.weight_dict
    gnd_loss = sum(gnd_loss_dict[k] * weight_dict[k] for k in gnd_loss_dict.keys() if k in weight_dict)
    
    gnd_loss = gnd_loss.to(torch.float32)
    
    total_loss = llm_loss + gnd_loss*0.1
    loss_dict = {'llm_loss': llm_loss, 'gnd_loss': gnd_loss, 'total_loss': total_loss}
    print(loss_dict)
    return total_loss, loss_dict

from utils.utils import ItemDataset
def create_dataset_function(image_processor, text_processor, cross_image_processor, grounding_image_processor, vg_token, path, args):
    dataset = ItemDataset(image_processor, text_processor, args, path, cross_image_processor=cross_image_processor, grounding_image_processor = grounding_image_processor,vg_token=vg_token)
    return dataset

from sat.model.finetune.lora2 import LoraMixin
from sat.model.finetune.prompt_tuning import PTuningV2Mixin

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--max_length', type=int)
    py_parser.add_argument('--ignore_pad_token_for_loss', action='store_false')
    py_parser.add_argument("--version", type=str, default="chat", choices=["chat", "vqa"], help='version to interact with')
    py_parser.add_argument("--from_pretrained", type=str, default="cogagent-chat", help='pretrained ckpt')
    py_parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    py_parser.add_argument("--vit_checkpoint_activations", action='store_true')
    py_parser = FineTuneTrainCogAgentModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    
    ####### Added As A Part Of The Fix ########
    with open('../model/cogagent-vqa/model_config.json', 'r') as f:
        missing_args = json.load(f)
    known.gnd_image_pix = missing_args['gnd_image_pix']
    known.eva_args = missing_args['eva_args']
    
    args = argparse.Namespace(**vars(args), **vars(known))
    
    print("args:\n", args, flush=True)
    if args.use_qlora:
        args.device = 'cpu'
        
    from utils.utils import llama2_tokenizer
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    # print the number of tokens
    # print(tokenizer.vocab_size) # 32000
    # print(tokenizer.pad_token_id, tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)) # 0, <unk>
    # print(tokenizer.bos_token_id, tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id)) # 1, <s>
    # print(tokenizer.eos_token_id, tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)) # 2, </s>
    # print(tokenizer.unk_token_id, tokenizer.convert_ids_to_tokens(tokenizer.unk_token_id)) # 0, <unk>
    
    #### ISSUE -> Add "[VG]" token to the tokenizer -> THIS IS NOT PROVIDED IN SAT LIBRARY
    # tokenizer.add_special_tokens({'additional_special_tokens': ["[VG]"]})
    # args.vg_token_idx = tokenizer.convert_tokens_to_ids("[VG]")
    # assert args.vg_token_idx == tokenizer.convert_tokens_to_ids("[VG]")
    
    ####### Temprorary Fix ########
    vg_token = "给"
    print("Using VG token: ", vg_token)
    args.vg_token_idx = tokenizer.convert_tokens_to_ids(vg_token)
    print("VG token index: ", args.vg_token_idx) # 31999
    assert args.vg_token_idx == tokenizer.convert_tokens_to_ids(vg_token)
    
    image_processor = get_image_processor(args.eva_args["image_size"][0])
    cross_image_processor = get_image_processor(args.cross_image_pix)
    grounding_image_processor = get_grounding_image_processor(args.gnd_image_pix)
    text_processor = llama2_text_processor(tokenizer, args.max_length, args.image_length)

    model, args = FineTuneTrainCogAgentModel.from_pretrained(args.from_pretrained, args, overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {})
    
    print("Model Created", flush=True)
    print("Model Size: ", sum(p.numel() for p in model.parameters())/1e6, "M", flush=True)
    
    # model, args = FineTuneTrainCogAgentModel.from_pretrained(args.from_pretrained, args, overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {})
    if args.use_ptuning: # TODO: wait for SAT updating
        model.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.pre_seq_len))

    if args.use_lora:
        model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range), reinit=True)
        model.get_mixin("eva").vit_model.add_mixin("lora", LoraMixin(args.eva_args['num_layers'], args.lora_rank, layer_range=args.layer_range), reinit=True)
    elif args.use_qlora:
        model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range, qlora=True), reinit=True)
        
    if args.use_qlora and torch.cuda.is_available():
        model = model.to('cuda')

    model = training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=partial(create_dataset_function, image_processor, text_processor, cross_image_processor,grounding_image_processor,vg_token), collate_fn=partial(data_collator, cross_image_processor=cross_image_processor))
    if args.use_lora:
        model.get_mixin("lora").merge_lora()
        model.get_mixin("eva").vit_model.get_mixin("lora").merge_lora()
        args.use_lora = False
        args.save = "checkpoints/merged_lora_cogagent"
        from sat.training.model_io import save_checkpoint
        save_checkpoint(1, model, None, None, args)