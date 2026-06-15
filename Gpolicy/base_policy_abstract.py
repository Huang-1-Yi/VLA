# -*- coding: utf-8 -*-
# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""BasePolicy —— 策略大脑(Fat Policy)统一基类。

v5-1 偏离版:从 A_common/types/base_policy.py 整体迁来,跟消费者
Gpolicy.PADP.padp_policy / Gpolicy.DP.dp_policy 同包,便于阅读和 IDE 跳转。

Gpolicy/__init__.py 通过 re-export 把这个接口暴露成
`from Gpolicy import BasePolicy` 这种最自然的写法。

继承 _apply 修复:policy.to(device) 时 normalizer 也会同步搬动(避免
normalizer 留在 CPU 而 model 跑到 GPU 引发 device mismatch)。
"""
from abc import abstractmethod
import torch
import torch.nn as nn


class BasePolicy(nn.Module):
    """所有策略大脑的统一基类(由 Gpolicy/<algo>/<algo>_policy.py 继承)。

    它是包含底层所有模型零件(Adapter/DM)以及策略调度逻辑(Scheduler/Loss)的超级模块。
    E_cti 只通过这个接口和整个系统交互。
    """

    def __init__(self):
        super().__init__()
        # 用 object.__setattr__ 让 _normalizer 不被 nn.Module 当成子模块
        # (避免 normalizer 进 state_dict、避免 .to(device) 时漏搬)
        object.__setattr__(self, "_normalizer", None)

    def _apply(self, fn, recurse=True):
        """override 以便 .to(device) / .cuda() / .cpu() 时把 normalizer 也同步搬动。"""
        super()._apply(fn, recurse)
        if self._normalizer is not None:
            # normalizer 是 LinearNormalizer(nn.Module),单独 _apply 一次
            object.__getattribute__(self, "_normalizer")._apply(fn)
        return self

    @abstractmethod
    def forward(self, a_t: torch.Tensor, t: torch.Tensor,
                global_cond: torch.Tensor) -> torch.Tensor:
        """单步 DM 封装(供 compute_loss / predict_action 内部调用)。

        子类内部实现:
            encoded_t = self.tm(t)
            return self.denoiser(a_t, global_cond=global_cond, encoded_t=encoded_t)
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, batch: dict) -> torch.Tensor:
        """接收 dataloader 吐出的 batch,执行前向传播和加噪,返回最终的标量 loss。

        E_cti 训练 loop 唯一入口:loss = policy.compute_loss(batch)
        """
        raise NotImplementedError

    @abstractmethod
    def predict_action(self, obs: dict):
        """接收实时 obs,执行 Denoise Loop 等推理逻辑,返回 ActionOutput 契约格式。

        E_cti eval/deploy 唯一入口:action_out = policy.predict_action(obs)
        """
        raise NotImplementedError

    def reset(self):
        """状态清空(默认无操作)。有状态 Policy(如 PADP 的 sliding window buffer)override。"""
        pass

    def set_normalizer(self, normalizer):
        """注入 normalizer(从数据集后算)。E_cti 推理前调一次。

        关键:normalizer **不进** state_dict(它独立保存/加载,不应该混进 policy 权重)。
        用 object.__setattr__ 绕过 nn.Module 的子模块自动注册。
        """
        object.__setattr__(self, "_normalizer", normalizer)
        object.__setattr__(self, "normalizer", normalizer)

    def shape_info(self) -> str:
        """返回 Policy 装配摘要(VM/AM/TM/DM 维度),供 __init__ 末尾 logger.info 用。"""
        return f"{type(self).__name__}"