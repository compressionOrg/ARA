from transformers import LlamaForCausalLM, LlamaConfig
from component.svd_linear import DynamicSVDLinear
from transformers import Qwen3ForCausalLM, Qwen3Config

class SVDLlamaConfig(LlamaConfig):
    model_type = "svd_llama"

class SVDQwen3Config(Qwen3Config):
    model_type = "svd_qwen3"


def set_nested_module(module, name: str, new_module):
    """
    根据路径字符串替换子模块
    """
    names = name.split(".")
    for n in names[:-1]:
        if n.isdigit():
            module = module[int(n)]
        else:
            module = getattr(module, n)
    setattr(module, names[-1], new_module)


class SVDQwen3ForCausalLM(Qwen3ForCausalLM):
    config_class = SVDQwen3Config

    def __init__(self, config):
        super().__init__(config)

        # 检查 config 里是否有需要替换的层
        if hasattr(config, "svd_linear_layers"):
            for key, layer_cfg in config.svd_linear_layers.items():
                # old_layer = get_nested_module(self, key)
                new_layer = DynamicSVDLinear.from_dict(layer_cfg)
                set_nested_module(self.model.layers, key, new_layer)
                
class SVDLlamaForCausalLM(LlamaForCausalLM):
    config_class = SVDLlamaConfig

    def __init__(self, config):
        super().__init__(config)

        # 检查 config 里是否有需要替换的层
        if hasattr(config, "svd_linear_layers"):
            for key, layer_cfg in config.svd_linear_layers.items():
                # old_layer = get_nested_module(self, key)
                if layer_cfg['static_bonuses'] is None or layer_cfg['finalized'] is False:
                    new_layer = DynamicSVDLinear.from_dict(layer_cfg)
                    set_nested_module(self.model.layers, key, new_layer)



