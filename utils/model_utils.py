#coding:utf8
import os
import sys
import torch
import torch.nn as nn

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# bandaid fix
dev = torch.device("cuda")

def get_model_from_huggingface(model_id):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
    from transformers import AutoConfig
    from component.dynamic_svd_llama import SVDLlamaConfig, SVDLlamaForCausalLM, SVDQwen3Config, SVDQwen3ForCausalLM
    AutoConfig.register("svd_llama", SVDLlamaConfig)
    AutoModelForCausalLM.register(SVDLlamaConfig, SVDLlamaForCausalLM)
    AutoConfig.register('svd_qwen3', SVDQwen3Config)
    AutoModelForCausalLM.register(SVDQwen3Config, SVDQwen3ForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # if "opt" in model_id or "mistral" in model_id:
    #     tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # else:
    #     tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if 'SVDLlama' in model_id:
        model = SVDLlamaForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True, cache_dir=None)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=None)
    model.seqlen = 2048
    return model, tokenizer

def get_model_from_local(model_id):
    pruned_dict = torch.load(model_id, weights_only=False, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    return model, tokenizer

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
