"""契约 6:BasePolicy —— 策略大脑(Fat Policy)统一基类。

铁律 3:本抽象类必须放在 A_common/types/。
Fat Policy 设计:
  - 子类 __init__ 实例化 Adapter + TM + DM
  - 子类 __init__ 一次性实例化 self.train_scheduler / self.infer_scheduler(若适用)
  - 子类 __init__ 缓存 self.horizon / self.action_dim
  - 子类暴露 3 个必须方法:forward + compute_loss + predict_action

E_cti 只通过这个接口和整个系统交互。
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