"""LinearNormalizer —— 单字段归一化(阶段 1 简化版)。

源:合并 PADP `diffusion_policy/common/normalize_util.py` 和
    `diffusion_policy/model/common/normalizer.py` 的核心,简化到只支持单字段。
"""
import json
import numpy as np
import torch
import torch.nn as nn


class SingleFieldLinearNormalizer(nn.Module):
    """单字段线性归一化:scale + offset,可用 stat 字典创建或加载。"""

    def __init__(self, scale: np.ndarray, offset: np.ndarray, input_stats_dict: dict = None):
        super().__init__()
        self.register_buffer("scale", torch.from_numpy(scale).float())
        self.register_buffer("offset", torch.from_numpy(offset).float())
        self.input_stats_dict = input_stats_dict or {}

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset) / (self.scale + 1e-8)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.offset

    @classmethod
    def create_manual(cls, scale, offset, input_stats_dict=None):
        return cls(scale=scale, offset=offset, input_stats_dict=input_stats_dict)

    @classmethod
    def create_from_stat(cls, stat: dict, output_min=-1.0, output_max=1.0):
        """从 {min, max, mean, std} 字典创建,目标范围 [output_min, output_max]。"""
        scale = (stat["max"] - stat["min"]) / (output_max - output_min) / 2
        offset = (stat["max"] + stat["min"]) / 2
        return cls(scale=scale, offset=offset, input_stats_dict=stat)

    @classmethod
    def create_identity(cls, dim: int):
        return cls(scale=np.ones(dim), offset=np.zeros(dim))


class LinearNormalizer(nn.Module):
    """多字段 normalizer 容器,按 key 索引子 normalizer。"""

    def __init__(self):
        super().__init__()
        self._modules: dict = {}

    def __setitem__(self, key: str, value: SingleFieldLinearNormalizer):
        self.add_module(key, value)

    def __getitem__(self, key: str) -> SingleFieldLinearNormalizer:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def normalize(self, data):
        out = {}
        for k, v in data.items():
            if k in self:
                out[k] = self[k].normalize(v)
            else:
                out[k] = v
        return out

    def unnormalize(self, data):
        out = {}
        for k, v in data.items():
            if k in self:
                out[k] = self[k].unnormalize(v)
            else:
                out[k] = v
        return out

    def load_state_dict(self, state_dict, strict: bool = True):
        """支持 dict 形式的 state_dict。"""
        # 跳过 nn.Module 默认行为,直接覆盖 _modules
        self._modules.clear()
        for k, v in state_dict.items():
            self[k] = v

    def state_dict(self, *args, **kwargs):
        return {k: getattr(self, k) for k in self._modules}

    def save(self, path: str):
        sd = {k: {"scale": v.scale.cpu().numpy(), "offset": v.offset.cpu().numpy(),
                  "input_stats_dict": v.input_stats_dict}
              for k, v in self._modules.items()}
        with open(path, "w") as f:
            json.dump(sd, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LinearNormalizer":
        n = cls()
        with open(path) as f:
            sd = json.load(f)
        for k, v in sd.items():
            n[k] = SingleFieldLinearNormalizer.create_manual(
                scale=np.asarray(v["scale"]),
                offset=np.asarray(v["offset"]),
                input_stats_dict=v.get("input_stats_dict", {}),
            )
        return n
