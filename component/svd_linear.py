import torch
import torch.nn as nn
import math
from typing import Union

class SVDLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ratio: float = 1, bias: bool = False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.ratio = ratio

        self.rank = int(self.in_features*self.out_features*ratio/(self.in_features+self.out_features))
        self.u_proj = nn.Linear(self.rank, self.out_features, bias=bias)
        self.v_proj = nn.Linear(self.in_features, self.rank, bias=False)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.u_proj(self.v_proj(x))
    
    def __repr__(self):
        return f"SVDLinear(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank})"


class DynamicSVDLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: Union[int, None] = None, bias: bool = False, ratio: float=1.0, finalized: bool=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.param_num = self.in_features * self.out_features
        self.ratio = ratio
        self.finalized = finalized
        self.retention_ratio = None

        if rank is not None:
            self.rank = rank
        else:
            self.rank = int(self.in_features*self.out_features*self.ratio/(self.in_features+self.out_features))
        self.u_proj = nn.Linear(self.rank, self.out_features, bias=bias)
        self.v_proj = nn.Linear(self.in_features, self.rank, bias=False)

        if not finalized:
            self.original_layer = nn.Linear(self.in_features, self.out_features, bias=False)

        self.static_bonuses = None

        self.retention_ratio_candidates = None
        self.retention_ratio_probabilities = None
        self.prob_map_matrix = None
        # self.register_buffer('final_mask', None)

    def update_ratio(self):
        self.retention_ratio_probabilities_softmax = self.retention_ratio_probabilities.softmax(dim=-1)
        self.retention_ratio = torch.matmul(self.retention_ratio_candidates, self.retention_ratio_probabilities_softmax)
        # return self.retention_ratio

    def init_learn_retention_ratio(self, D=100, target_ratio: float = None):
        dev = self.u_proj.weight.device
        # 候选列表：1.0, 0.99, ..., 0.0
        dtype = self.u_proj.weight.dtype
        retention_ratio_step = 1/D
        self.retention_ratio_candidates = torch.arange(
            1.0, -1 * retention_ratio_step, -1 * retention_ratio_step, device=dev
        )
        self.retention_ratio_candidates[-1] = 0.0

        # 如果不指定 target_ratio，就默认均匀初始化（期望≈0.5）
        if target_ratio is None:
            print('misssing target ratio parameter')
        else:
            target_ratio = target_ratio/self.ratio
            c = self.retention_ratio_candidates
            # 分组：大于等于 target_ratio 和小于 target_ratio
            A = c >= target_ratio
            B = ~A
            n1, n2 = A.sum().item(), B.sum().item()
            S1, S2 = c[A].sum().item(), c[B].sum().item()

            # 计算 logit 差值 Δ = p1 - p2
            t = (S2 - target_ratio * n2) / (target_ratio * n1 - S1)
            delta = math.log(t)

            # 构造 logits：A组 logits=delta，B组 logits=0
            logits = torch.zeros_like(c)
            logits[A] = delta
            logits[B] = 0.0

            self.retention_ratio_probabilities = nn.Parameter(logits.to(device=dev))

        # 构造 prob_map_matrix
        self.prob_map_matrix = torch.zeros((len(self.retention_ratio_candidates), self.rank), device=dev)
        for i in range(len(self.retention_ratio_candidates)):
            self.prob_map_matrix[i, :int(self.rank * self.retention_ratio_candidates[i].item())] = 1

    
    # def finish_learn_retention_ratio(self, factor):
    #     # self.update_ratio()
    #     dev = self.u_proj.weight.device
    #     dtype = self.u_proj.weight.dtype
    #     final_ratio = self.retention_ratio*factor
    #     final_ratio = max(final_ratio, 0.05)
    #     self.final_mask = torch.zeros(self.rank, device=dev, dtype=dtype)
    #     self.final_mask[:int(final_ratio*self.rank)] = 1

    #     self.retention_ratio_candidates = None
    #     self.retention_ratio_probabilities = None
    #     self.prob_map_matrix = None

    def finish_learn_retention_ratio(self, factor: float):
        """
        Permanently prunes the layer by resizing the u_proj and v_proj weight matrices.
        This eliminates the need for a mask during the forward pass.
        """
        dev = self.u_proj.weight.device
        dtype = self.u_proj.weight.dtype

        # 1. Calculate the new, reduced rank
        final_ratio = self.retention_ratio * factor
        # if final_ratio < 0.01:
        #     print(f'final ratio is {final_ratio}, adjust to 0.01')
        #     final_ratio = 0.01
        
        new_rank = int(final_ratio * self.rank)

        # Ensure new_rank is at least 1 to avoid creating empty layers
        if new_rank < 1:
            new_rank = 1

        # 2. Create new, smaller linear layers
        new_v_proj = nn.Linear(self.in_features, new_rank, bias=False, device=dev, dtype=dtype)
        new_u_proj = nn.Linear(new_rank, self.out_features, bias=(self.u_proj.bias is not None), device=dev, dtype=dtype)

        # 3. Copy the weights and biases that are being kept
        # For v_proj, we keep the first `new_rank` rows of its weight matrix
        new_v_proj.weight.data.copy_(self.v_proj.weight.data[:new_rank, :])
        
        # For u_proj, we keep the first `new_rank` columns of its weight matrix
        new_u_proj.weight.data.copy_(self.u_proj.weight.data[:, :new_rank])
        
        if self.u_proj.bias is not None:
            new_u_proj.bias.data.copy_(self.u_proj.bias.data)

        # 4. Replace the old layers with the new, pruned layers
        self.v_proj = new_v_proj
        self.u_proj = new_u_proj

        self.original_layer = None 
        self.static_bonuses = None

        # 5. Update the layer's rank and clean up
        self.ratio = final_ratio.item() * self.ratio
        self.rank = new_rank
        self.retention_ratio_candidates = None
        self.retention_ratio_probabilities = None
        self.prob_map_matrix = None

        self.finalized = True

    def get_hard_mask(self):
        dev = self.u_proj.weight.device
        hard_mask = torch.zeros(self.rank, device=dev)
        hard_mask[:int(torch.round(self.rank*self.retention_ratio).item())] = 1

        return hard_mask

    def get_prob_mask(self):
        dev = self.u_proj.weight.device
        # prob_mask = torch.zeros(self.rank, device=dev)
        prob_mask = (self.retention_ratio_probabilities_softmax @ self.prob_map_matrix).to(dev)
        
        return prob_mask

    def get_sigma_mask(self):
        hard_mask = self.get_hard_mask()
        prob_mask = self.get_prob_mask()
        sigma_mask = hard_mask.detach() - prob_mask.detach() + prob_mask
        sigma_mask = sigma_mask.to(self.u_proj.weight.dtype)

        return sigma_mask

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.finalized:
            return self.u_proj(self.v_proj(x))
        else:
            if self.retention_ratio is None or self.retention_ratio*self.ratio < 1:
                self.update_ratio()
                mask = self.get_sigma_mask()
                return self.u_proj(self.v_proj(x)*mask)
            else: 
                return self.original_layer(x)
            
    def __repr__(self):
        return f"DynamicSVDLinear(in_features={self.in_features}, out_features={self.out_features}, retention_ratio={self.retention_ratio}, finalized={self.finalized})"

    def to_dict(self):
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.u_proj.bias is not None,
            "rank": self.rank,
            "ratio": self.ratio,
            "static_bonuses": self.static_bonuses,
            "finalized": self.finalized
        }
    
    @classmethod
    def from_dict(cls, cfg:dict):
        instance = cls(
            in_features=cfg["in_features"],
            out_features=cfg["out_features"],
            bias=cfg["bias"],
            rank=cfg["rank"],
            ratio = cfg["ratio"],
            finalized = cfg.get("finalized", False),
        )
        instance.static_bonuses = cfg.get("static_bonuses", None)

        return instance
    
def init_learn_retention_ratio(svd_linears, D=100, target_ratio: float=None):
    for layer_name in svd_linears:
        svd_linear = svd_linears[layer_name]
        svd_linear.init_learn_retention_ratio(D=D, target_ratio=target_ratio)

def get_retention_ratio(svd_linears:list[DynamicSVDLinear]):
    total_param_num = 0
    weighted_clamped_ratio_sum = 0
    for layer in svd_linears.values():
        r_unclamped = layer.ratio * layer.retention_ratio
        r_clamped = min(r_unclamped.item(), 1)

        # 累加用于计算 L_compression 的部分
        total_param_num += layer.param_num
        weighted_clamped_ratio_sum += r_clamped * layer.param_num
    R_effective_total = weighted_clamped_ratio_sum / total_param_num

    return R_effective_total


def get_bonus_for_ratio(bonuses: dict, r_eff: torch.Tensor):
    """根据有效比率，从字典中查找最近的 bonus 值。"""
    if not bonuses:
        return 0.0
    # 将 tensor 转换为 float 进行比较
    ratio_val = r_eff.item()
    # 找到字典中与当前 ratio 最接近的 key
    closest_key = min(bonuses.keys(), key=lambda k: abs(float(k) - ratio_val))
    return bonuses[closest_key]

def calculate_regularization_loss(svd_linears: dict, target_ratio: float, alpha: float, beta: float):
    """计算 L_compression 和 L_bonus 组成的正则化总损失。"""
    total_param_num = 0
    weighted_clamped_ratio_sum = 0
    total_bonus_loss = 0
    
    dev = 'cuda'
    # 确保至少有一个 svd_linear 层
    if not svd_linears:
        return torch.tensor(0.0)


    for layer in svd_linears.values():
        ## 防止在使用满秩结构时计算图的分离报错
        probabilities_softmax = layer.retention_ratio_probabilities.softmax(dim=-1)
        current_retention_ratio = torch.matmul(layer.retention_ratio_candidates, probabilities_softmax)
        
        r_unclamped = layer.ratio * current_retention_ratio
        r_clamped = torch.min(r_unclamped, torch.tensor(1.0, device=r_unclamped.device))

        r_clamped = r_clamped.to(device=dev)
        # 累加用于计算 L_compression 的部分
        total_param_num += layer.param_num
        weighted_clamped_ratio_sum += r_clamped * layer.param_num
        
        # 累加用于计算 L_bonus 的部分
        bonus = get_bonus_for_ratio(layer.static_bonuses, r_clamped)
        layer_bonus_loss = (1 - r_clamped)*(r_clamped>(1-bonus))
        total_bonus_loss += layer_bonus_loss

    # 计算 L_compression
    R_effective_total = weighted_clamped_ratio_sum / total_param_num
    L_compression = (R_effective_total - target_ratio)**2
    # L_compression = abs(R_effective_total - target_ratio)
    
    # 计算 L_bonus
    L_bonus = total_bonus_loss / len(svd_linears)
    
    return alpha * L_compression + beta * L_bonus

def finalize_model(model, svd_linears: dict, target_ratio):
    layers_to_replace = {}
    layers_to_compress = {}
    ratios_dict = {}
    total_params = variable_params = 0
    for name in svd_linears:
        svd_linear = svd_linears[name]
        svd_linear.update_ratio()
        layer_ratio = (svd_linear.retention_ratio.item())*svd_linear.ratio

        ratios_dict[name] = layer_ratio
        total_params += svd_linear.param_num
        
        if layer_ratio >= 0.99:
            layers_to_replace[name] = svd_linear
            svd_linear.retention_ratio = svd_linear.retention_ratio/svd_linear.retention_ratio
            last_name = name.split('.',2)[2]
            del model.config.svd_linear_layers[last_name]
        else:
            layers_to_compress[name] = svd_linear
            variable_params += svd_linear.param_num * layer_ratio
    
    torch.save(ratios_dict, 'layer_ratio.pt')
    retention_ratio = get_retention_ratio(svd_linears)

    
    factor = 1 - (retention_ratio - target_ratio)*total_params/variable_params if variable_params>0 else 1

    for name in layers_to_compress:
        svd_linear = layers_to_compress[name]
        svd_linear.finish_learn_retention_ratio(factor)

    # 执行替换操作
    if layers_to_replace:
        print("Replacing layers with their original full-rank versions...")
        # 需要一个 set_nested_module 的辅助函数来根据名字设置模块
        # (假设此函数已存在)
        for name in layers_to_replace:
            svd_linear = layers_to_replace[name]
            original_layer = svd_linear.original_layer
            parent_module_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_module_name)
            setattr(parent_module, child_name, original_layer)
    
    print("Model finalization complete. ✅")

def final_learn_retention_ratio(svd_linears, target_ratio):
    for name in svd_linears:
        svd_linear = svd_linears[name]
        svd_linear.update_ratio()
        print(f'{name} retention ratio {svd_linear.retention_ratio*svd_linear.ratio}')
        # if svd_linear.retention_ratio*svd_linear.ratio >= 1:
        #     print(f'{name} retention ratio {svd_linear.retention_ratio*svd_linear.ratio}')
    retention_ratio = get_retention_ratio(svd_linears)
    # print(f'final learned retention ratio: {retention_ratio}')
    factor = target_ratio/retention_ratio
    for name in svd_linears:
        svd_linear = svd_linears[name]
        svd_linear.finish_learn_retention_ratio(factor)

def get_retention_ratio_params(svd_linears):
    params = []
    for name in svd_linears:
        svd_linear = svd_linears[name]
        if svd_linear.retention_ratio_probabilities is not None:
            layer_retention_ratio_params = svd_linear.retention_ratio_probabilities
            if type(layer_retention_ratio_params) is list:
                params.extend(layer_retention_ratio_params)
            else:
                params.append(layer_retention_ratio_params)
    return params

def update_svd_config_before_saving(model):
    """
    遍历模型，找到所有 DynamicSVDLinear 层，并使用它们的最新状态
    更新 model.config.svd_linear_layers 中的配置。
    """
    if not hasattr(model.config, "svd_linear_layers"):
        print("警告: 在 model.config 中未找到 svd_linear_layers。")
        return

    # 确定模型中 transformer 层的基本路径
    layer_prefix = ""
    if hasattr(model.model, 'layers'):  # 适用于 LLaMA, Mistral 等模型
        layer_prefix = "model.layers"
    elif hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):  # 适用于 OPT 等模型
        layer_prefix = "model.decoder.layers"
    else:
        print("错误: 无法确定模型的 layer 前缀路径。")
        return

    print("正在刷新 model.config.svd_linear_layers...")
    for full_name, module in model.named_modules():
        if isinstance(module, DynamicSVDLinear):
            # 将模块的完整名称转换为存储在 config 中的 key
            # 例如: "model.layers.15.mlp.gate_proj" -> "15.mlp.gate_proj"
            if full_name.startswith(layer_prefix):
                # +1 是为了去掉前缀后面的那个点 '.'
                config_key = full_name[len(layer_prefix) + 1:] 
                
                if config_key in model.config.svd_linear_layers:
                    # 从模块实例获取【当前最新】的配置字典
                    current_config = module.to_dict()
                    # 更新 model.config 中的信息
                    model.config.svd_linear_layers[config_key] = current_config
                else:
                    print(f"警告: 在 svd_linear_layers 配置中找不到 key: {config_key}")
    
    print("model.config.svd_linear_layers 已成功刷新。")