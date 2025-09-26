#coding:utf8
import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn
from timm.utils import NativeScaler

from utils.data_utils import *
# from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
# from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
# from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import *
from evaluater import * 
# from component.dynamic_svd_llama import Dynamic_SVD_LlamaMLP, Dynamic_SVD_LlamaAttention
from component.svd_linear import *
from optimizers import Prodigy

from transformers import AutoConfig, AutoModelForCausalLM
from component.dynamic_svd_llama import SVDLlamaConfig, SVDLlamaForCausalLM

AutoConfig.register("svd_llama", SVDLlamaConfig)
AutoModelForCausalLM.register(SVDLlamaConfig, SVDLlamaForCausalLM)


current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        # for name, module in model.named_modules():
            # if 'rotary_emb' in name:
            #     module.inv_freq = module.inv_freq.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs.get('position_ids', None)
                cache['position_embeddings'] = kwargs.get(
                    'position_embeddings', None)
            else:
                cache['attention_mask'] = torch.cat(
                    (cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if kwargs.get('position_ids', None) is not None:
                    cache['position_ids'] = torch.cat(
                        (cache['position_ids'], kwargs['position_ids']), dim=0)

            raise ValueError
            
    layers[0] = Catcher(layers[0])

    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)

    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    profiling_mat = {}

    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))

        for j in range(inps.shape[0]):
            if position_ids is None:
                outs[j] = layer(inps[j].unsqueeze(
                    0), attention_mask=attention_masks[j].unsqueeze(0))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks, position_ids=position_ids[0].unsqueeze(
                    0), position_embeddings=position_embeddings)[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            layer_profile[name] = subset[name].scaling_diag_matrix.cpu()
            subset[name].scaling_diag_matrix = None
            del subset[name].scaling_diag_matrix

        torch.cuda.empty_cache()

        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat
        

# @torch.no_grad()
# def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
#     if "opt" in model_name:
#         layers = model.model.decoder.layers
#         model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
#         model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
#         model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
#     else:
#         layers = model.model.layers
#         model.model.embed_tokens = model.model.embed_tokens.to(dev)
#         model.model.norm = model.model.norm.to(dev)
#     layers[0] = layers[0].to(dev)

#     dtype = next(iter(model.parameters())).dtype
#     inps = torch.zeros(
#         (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
#     )
#     cache = {'i': 0, 'attention_mask': None, "position_ids": None}
#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module
#         def forward(self, inp, **kwargs):
#             inps[cache['i']] = inp.cpu()
#             cache['i'] += 1
#             if cache['attention_mask'] is None:
#                 cache['attention_mask'] = kwargs['attention_mask'].cpu()
#                 if "opt" not in model_name:
#                     cache['position_ids'] = kwargs['position_ids'].cpu()
#             else:
#                 cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)
#                 if "opt" not in model_name:
#                     cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
#             raise ValueError
#     layers[0] = Catcher(layers[0])
#     for batch in calib_loader:
#         try:
#             batch = {k: v.to(dev) for k, v in batch.items()}
#             model(**batch)
#         except ValueError:
#             pass
#     layers[0] = layers[0].module
#     layers[0] = layers[0].cpu()
#     if "opt" in model_name:
#         model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
#         model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
#         model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
#     else:  
#         model.model.embed_tokens = model.model.embed_tokens.cpu()
#         model.model.norm = model.model.norm.cpu()
#     torch.cuda.empty_cache()
#     outs = torch.zeros_like(inps)
#     attention_masks = cache['attention_mask']
#     if "opt" not in model_name:
#         position_ids = cache['position_ids']
#     profiling_mat = {}
#     for i in tqdm(range(len(layers))):
#         layer_profile = {}
#         layer = layers[i].to(dev)
#         subset = find_layers(layer)        
#         def hook(module, input, output):
#             inp = input[0].detach().float()
#             if inp.dim() == 2:  # for opt
#                 inp = inp.unsqueeze(0)
#             adds = torch.matmul(inp.transpose(1,2), inp)
#             adds_sum = torch.sum(adds, dim=0)
#             module.scaling_diag_matrix += adds_sum
#             del inp, adds, adds_sum, output
#             torch.cuda.empty_cache()
#         handles = []
#         for name in subset:
#             subset[name].scaling_diag_matrix = 0
#             handles.append(subset[name].register_forward_hook(hook))
#         for j in range(inps.shape[0]):
#             if "opt" not in model_name:
#                 outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
#             else:
#                 outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
#         for h in handles:
#             h.remove()
#         layer = layer.cpu()
#         for name in subset:
#             subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
#         torch.cuda.empty_cache()
#         for name in subset:
#             raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
#             try:
#                 scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
#             except Exception as e:
#                 print("Warning: eigen scaling_diag_matrix is not positive!")
#                 eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
#                 raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
#                 scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
#                 eigenvalues = None
#                 del eigenvalues
#             layer_profile[name] = scaling_diag_matrix.cpu()
#             scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
#             del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
#             torch.cuda.empty_cache()
#         layers[i] = layer.cpu()
#         profiling_mat[i] = layer_profile
#         inps = outs
#         torch.cuda.empty_cache()
#     return profiling_mat



@torch.no_grad()
def svd_llm(W, raw_scaling_diag_matrix, ratio):
    raw_scaling_diag_matrix = raw_scaling_diag_matrix.double()
    try:
        scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
    except Exception as e:
        print("Warning: eigen scaling_diag_matrix is not positive!")
        # eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
        # raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
        damp = 0.05 * torch.mean(torch.diag(raw_scaling_diag_matrix))
        n = raw_scaling_diag_matrix.size(0)
        idx = torch.arange(n, device=raw_scaling_diag_matrix.device)
        raw_scaling_diag_matrix[idx, idx] += damp
        scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
        eigenvalues = raw_scaling_diag_matrix = None
        del eigenvalues
    try:
        scaling_diag_matrix = scaling_diag_matrix.float()
        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
    except Exception as e:
        print("Warning: scaling_diag_matrix is not full rank!")
        scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
    scaling_diag_matrix = scaling_diag_matrix.float()
    scaling_matrix_inv = scaling_matrix_inv.float()
    W_scale = torch.matmul(W, scaling_diag_matrix)
    U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
    num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
    truc_s = S[:num_s_after_trunc]
    truc_u = U[:, :num_s_after_trunc]
    truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
    truc_sigma = torch.diag(truc_s)
    #### Replace Attn, MLP ####
    sqrtSigma = torch.sqrt(truc_sigma)
    svd_u = torch.matmul(truc_u, sqrtSigma)
    svd_v = torch.matmul(sqrtSigma, truc_v)
    return svd_u, svd_v, S

def cal_bonus(in_features, out_features, S):
    bonuses = {}
    total_loss = torch.sum(S**2)
    
    # Loop over integers from 1 to 99
    for i in range(1, 100):
        # Convert the integer to a float ratio (0.01, 0.02, ..., 0.99)
        ratio_step = i / 100.0
        
        k = int(in_features * out_features * ratio_step / (in_features + out_features))
        truncated_loss = torch.sum(S[k:]**2)
        bonus = torch.sqrt(truncated_loss / total_loss).item()
        
        # Using round() for the key is good practice to ensure consistency
        bonuses[round(ratio_step, 2)] = bonus
    
    return bonuses

def set_nested_module(module: nn.Module, name: str, new_sub_module: nn.Module):
    """
    根据点分隔的名称路径，在模块中设置一个嵌套的子模块。

    例如: set_nested_module(layer, 'self_attn.q_proj', new_q_proj)
        这等效于 layer.self_attn.q_proj = new_q_proj
    """
    keys = name.split('.')
    # 遍历路径，直到倒数第二个key，以获取直接的父模块
    parent = module
    for key in keys[:-1]:
        parent = getattr(parent, key)
    
    # 在父模块上设置最后一个key对应的属性为新的子模块
    setattr(parent, keys[-1], new_sub_module)

@torch.no_grad()
def linear_whitening(model_name, model, profiling_mat, ratio, dev):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    if not hasattr(model.config, "svd_linear_layers"):
        model.config.svd_linear_layers = {}
    for i, layer in enumerate(tqdm(layers)):
        subset = find_layers(layer)
        for name in subset:
            # 1. 获取原始层及其配置
            original_linear = subset[name]
            in_features = original_linear.in_features
            out_features = original_linear.out_features
            has_bias = original_linear.bias is not None
            dtype = original_linear.weight.dtype
            device = 'cpu'

            W = original_linear.weight.data.float()
            raw_scaling_diag_matrix = profiling_mat[i][name].to(W.device)
            # svd_linear = SVDLinear(in_features, out_features, ratio, bias=has_bias).to(device).to(dtype)
            # svd_u, svd_v = svd_llm(W, raw_scaling_diag_matrix, ratio)

            svd_linear = DynamicSVDLinear(in_features, out_features, bias=has_bias, ratio=args.ratio).to(device).to(dtype)
            svd_u, svd_v, S = svd_llm(W, raw_scaling_diag_matrix, args.ratio)

            torch.save(S, 'S.pt')
            bonuses = cal_bonus(in_features, out_features, S)

            svd_linear.static_bonuses = bonuses
            svd_linear.u_proj.weight.data.copy_(svd_u.to(device))
            svd_linear.v_proj.weight.data.copy_(svd_v.to(device))
            svd_linear.original_layer.weight.data.copy_(W.to(device))
            if has_bias:
                svd_linear.u_proj.bias.data.copy_(original_linear.bias.data)
            
            
            set_nested_module(layer, name, svd_linear)
            model.config.svd_linear_layers[f"{i}.{name}"] = svd_linear.to_dict()
            W = raw_scaling_diag_matrix = svd_u = svd_v = None
            torch.cuda.empty_cache()
    if 'Llama' in model_name or 'llama' in model_name:
        model.config.model_type = 'svd_llama'
    elif 'Qwen3' in model_name:
        model.config.model_type = 'svd_qwen3'
    print("\nReplacement complete.")
    return model

def training_mask(model, tokenizer, args):
    svd_linears = find_layers(model, layers=[DynamicSVDLinear])
    model.requires_grad_(False)
    
    init_learn_retention_ratio(svd_linears, D=args.D, target_ratio=args.ratio)
    for m in svd_linears.values():
        m.retention_ratio_probabilities.requires_grad_(True)
    retention_ratio_params = get_retention_ratio_params(svd_linears)
    loss_scaler = NativeScaler()
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(retention_ratio_params, weight_decay=0, lr=args.lr)
    elif args.optimizer == 'prodigy':
        optimizer = Prodigy(retention_ratio_params, d_coef=args.d_coef)
    else: 
        print('optimizer type error, please choose from adamw and prodigy')
    if 'alpaca' in args.train_dataset:
        train_loader = get_train_loader_for_alpaca(args.train_dataset, tokenizer, args.train_seq_len, args.train_batch_size, args.train_nsamples, args.seed)
    else:
        train_loader = get_train_loader_for_causal_lm(args.train_dataset, tokenizer, args.train_seq_len, args.train_batch_size, args.train_nsamples, args.seed)
    device = args.DEV
    loss_list, model_loss_list, reg_loss_list = [], [], []
    for epoch in range(args.epochs):
        epoch_loss_list, epoch_model_loss_list, epoch_reg_loss_list = [], [], []
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        for batch in tqdm(train_loader):
            # 将数据移动到指定设备
            if batch['input_ids'].ndim == 3:
                batch['input_ids'] = batch['input_ids'].squeeze(0)
                batch['labels'] = batch['labels'].squeeze(0)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, labels=labels)
            model_loss = outputs.loss


            # alpha_current = args.alpha*max(1, (epoch+1)/5)
            alpha_current = args.alpha
            reg_loss = calculate_regularization_loss(
                svd_linears, 
                target_ratio=args.ratio, 
                alpha=alpha_current, 
                beta=args.beta 
            )
            total_loss = model_loss + reg_loss

            loss_scaler(total_loss, optimizer, parameters=retention_ratio_params)

            epoch_loss_list.append(total_loss.item())
            epoch_model_loss_list.append(model_loss.item())
            epoch_reg_loss_list.append(reg_loss.item())

        loss_list.append(epoch_loss_list)
        model_loss_list.append(epoch_model_loss_list)
        reg_loss_list.append(epoch_reg_loss_list)
        
        retention_ratio = get_retention_ratio(svd_linears)
        print(f'retention ratio: {retention_ratio}')
        print(f'mean model loss: {sum(epoch_model_loss_list)/len(epoch_model_loss_list)}')

        if args.epochs_ppl:
            ppl_eval(model, tokenizer, datasets=['wikitext2', 'c4'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
    model.requires_grad_(False)

    finalize_model(model, svd_linears, args.ratio)
    update_svd_config_before_saving(model)
    torch.cuda.empty_cache()
    ppl_eval(model, tokenizer, datasets=['wikitext2', 'c4'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
    task_list=["mathqa", "piqa","hellaswag", "winogrande", "arc_easy","arc_challenge", "openbookqa"]
    zero_shot_eval(args.model, model, tokenizer, task_list)



@torch.no_grad()
def heuristic_mask(model, tokenizer, args):

    svd_linears = find_layers(model, layers=[DynamicSVDLinear])
    if args.ratio_method == 'uniform':
        init_learn_retention_ratio(svd_linears, D=args.D, target_ratio=args.ratio)
        finalize_model(model, svd_linears, args.ratio)
        update_svd_config_before_saving(model)
    elif args.ratio_method == 'ars':
        import json
        with open(args.ars_path, 'r') as f:
            all_layer_data = json.load(f)
        for layer_data in all_layer_data:
            layer_name = layer_data['layer_name']
            if 'gate' in layer_name or 'up' in layer_name or 'down' in layer_name:
                layer_name = f"model.layers.{layer_data['layer_idx']}.mlp.{layer_data['layer_name']}"
            else:
                layer_name = f"model.layers.{layer_data['layer_idx']}.self_attn.{layer_data['layer_name']}"
            layer_ratio = layer_data['param_ratio']
            layer_ratio = min(layer_ratio, 1.1)
            svd_linear = {layer_name:svd_linears[layer_name]}
            init_learn_retention_ratio(svd_linear, D=args.D, target_ratio=layer_ratio)
        finalize_model(model, svd_linears, args.ratio)
        update_svd_config_before_saving(model)
    torch.cuda.empty_cache()
    ppl_eval(model, tokenizer, datasets=['wikitext2','c4'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
    # task_list=["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
    task_list=["mathqa", "piqa","hellaswag", "winogrande", "arc_easy","arc_challenge", "openbookqa"]
    zero_shot_eval(args.model, model, tokenizer, task_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='/cluster/home/xulin/programs/models/Llama-2-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default='compressed_models/SVDLlama-2-13b-hf_1.2', help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.8, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--D', type=float, default=100)
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str,default='test', help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=3, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--train_dataset', type=str, default='c4',help='training data from [wikitext2, ptb, c4, alpaca-cleaned/alpaca_data_cleaned.json]')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--train_nsamples', type=int, default=256, help='Number of calibration data samples for training.')
    parser.add_argument('--train_seq_len', type=int, default=512, help='the default train sequence length of the LLM')
    parser.add_argument('--train_batch_size', type=int, default=1, help='train bactch size')
    parser.add_argument('--alpha', type=float, default=1e2, help='rate of ratio loss')
    parser.add_argument('--beta', type=float, default=5e1, help='rate of full rank loss')
    parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer type')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for adamw')
    parser.add_argument('--d_coef', type=float, default=1e-2, help='d coef for prodigy')
    parser.add_argument('--epochs_ppl', action='store_true', default=False, help='ppl eval every training epoch')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--ratio_method', type=str, default='ars')
    parser.add_argument('--ars_path', type=str)
    parser.add_argument('--step', type=int, default=0, help='the step to run the compression')
    parser.add_argument('--cuda_devices', type=str, default='7', help='the cuda devices to run the model')
    

    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    # args.ratio = 1- args.ratio
    if args.step == 0:
        # model, tokenizer = get_model_from_local(args.model_path)
        # model = model.to('cuda')
        model, tokenizer = get_model_from_huggingface(model_id=args.model_path)
        heuristic_mask(model, tokenizer, args)
        # if args.save_path is not None:
        #     save_dir = args.save_path + '/' + args.model.split('/')[-1] +'_' + args.ratio_method + '_' + str(args.ratio)
        #     model.save_pretrained(save_dir)
        #     tokenizer.save_pretrained(save_dir)
    elif args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        if args.profiling_mat_path is None:
            model = model.to('cpu')
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.split('/')[-1] + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
                # torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
            exit()
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        model = linear_whitening(args.model, model, profiling_mat, args.ratio, args.DEV)
        model = model.half()
        if args.save_path is not None:
            save_dir = args.save_path + '/' + 'SVD' + args.model.split('/')[-1] + '_' + str(args.ratio)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    elif args.step == 2:
        # model, tokenizer = get_model_from_local(args.model_path)
        # model = model.to('cuda')
        model, tokenizer = get_model_from_huggingface(model_id=args.model_path)
        training_mask(model, tokenizer, args)
        if args.save_path is not None:
            model_name = args.model_path.split('/')[-1]
            save_dir = os.path.join(args.save_path, f"{model_name}_whitening_training_{args.ratio}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    elif args.step == 3:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        # task_list = ["boolq", "rte", "hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        task_list = ["hellaswag","winogrande", "arc_easy", "arc_challenge", "openbookqa", "piqa", "mathqa"]
        # task_list = ['mathqa', 'piqa']

        model, tokenizer = get_model_from_huggingface(args.model_path)
        ppl_eval(model, tokenizer, datasets=['wikitext2', 'c4'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        # ppl_eval(model, tokenizer, datasets=['wikitext2', 'c4'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        hflm = HFLM(pretrained=model, tokenizer=tokenizer)
        # hflm = HFLM(pretrained=args.model_path, tokenizer=args.model_path, dtype=torch.float16)
        res = lm_eval.simple_evaluate(hflm, tasks=task_list, num_fewshot=0)
        print(res['results'])
    elif args.step == 4:
        model, tokenizer = get_model_from_huggingface(args.model_path)
        # eff_eval(model, tokenizer, generated_len=32, batch_size=32, device=args.DEV)
        eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
