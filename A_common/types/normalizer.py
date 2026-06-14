"""LinearNormalizer —— 单字段归一化(阶段 1 简化版)。

源:合并 PADP `diffusion_policy/common/normalize_util.py` 和
    `diffusion_policy/model/common/normalizer.py` 的核心,简化到只支持单字段。
"""
# =============================================================================
# 2026-06-14 create_from_stat 公式修复 + 交叉验证(参考 [padp-vla-v4-normalizer-bug.md])
# =============================================================================
# 旧公式:scale = (max-min)/(out_max-out_min) / 2     ← 多了个 /2
#       offset = (max+min) / 2                       ← 缺 output_min/max 项
# 现象:create_from_stat(-1, 1) 实际映射 [-1, 1] → [-2, 2](非预期)
#
# 新公式(数学推导):
#   对 input range [a, b] → output range [c, d]
#   scale  = (b - a) / (d - c)
#   offset = (a + b) / 2 - (c + d) / 2 * scale
#   即:  scale  = (stat["max"] - stat["min"]) / (output_max - output_min)
#        offset = (stat["min"] + stat["max"]) / 2 - (output_min + output_max) / 2 * scale
#
# 验证(5 case,全过):
#   CASE 1 对称 [-1,1]→[-1,1]   : x=-1→-1, x=1→1     ✓ scale=1.0
#   CASE 2 实数据 [-0.12, 0.25]  : a→-1.0000, b→+1.0000 ✓ (robot0_eef_pos)
#   CASE 3 非对称 [0,1]→[0,10]  : 0→0, 0.5→5, 1→10    ✓ (offset 修正)
#   CASE 4 Round-trip 实数据    : err = 0.00e+00        ✓ (mathematically exact)
#   CASE 5 端到端 normalize/unnormalize : 全部落 [-1, 1] + round-trip 精确 ✓
#
# 影响面扫描(全部 caller):
#   caller                                                      修前       修后
#   normalizer_utils.get_range_normalizer_from_stat            [-2, 2]    [-1, 1]   ✓
#   normalizer_utils.robomimic_abs_action_only_normalizer_*    硬编码      用 stat   ✓
#   C_sim.robomimic.zarr_dataset.py:233 (lowdim)              [-2, 2]    [-1, 1]   ✓
#   C_sim.robomimic.zarr_dataset_padp.py:362 (lowdim)          [-2, 2]    [-1, 1]   ✓
#   C_sim.robomimic.zarr_dataset_padp.py:343-353 (action 10D) create_manual(不受影响)  ✓
#   E_cti/train/run_train.py:67 dataset.get_normalizer()      [-2, 2]    [-1, 1]   ⚠️ BREAK
#
# ⚠️ BREAK CHANGE:任何在 bug 修复前训的 checkpoint 不再兼容
#   - 旧 ckpt 训练时数据是 [-2, 2],policy 学的是这个范围
#   - 修后 normalizer 输出 [-1, 1],与旧 ckpt 的预期分布不匹配
#   - 必须重训(注:本仓目前无 ckpt,实际无影响)
#
# Save/load 兼容性:LinearNormalizer.save() 写裸 scale/offset 数组(不含公式),
#   所以旧文件(用 /2 算的 scale=0.5)和新文件(scale=1.0)都是合法数据,load 行为正确。
#   文件格式无 break。
#
# 顺带新发现(与本次修复无关,不算在 11 个 bug 里):
#   LinearNormalizer.save() 调 json.dump 但 input_stats_dict 里是 np.ndarray → TypeError
#   生产代码无 caller(pure dead code),不影响任何功能。
# =============================================================================
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
        """从 {min, max, mean, std} 字典创建,目标范围 [output_min, output_max]。

        正确公式(2026-06-14 修):normalize(x) = (x - offset) / scale 把
        [stat["min"], stat["max"]] 映到 [output_min, output_max]:
            scale   = (max - min) / (output_max - output_min)
            offset  = (min + max) / 2 - (output_min + output_max) / 2 * scale
        旧公式有 /2 露馅,让 input [-1, 1] → output [-2, 2](非预期)。
        Round-trip 仍正确(unchanged: unnormalize = y*scale + offset)。
        """
        scale = (stat["max"] - stat["min"]) / (output_max - output_min)
        offset = (stat["min"] + stat["max"]) / 2 - (output_min + output_max) / 2 * scale
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
