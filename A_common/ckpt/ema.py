# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""EMAModel —— 指数移动平均(从 PADP 直接抄)。

源:`../../PADP/diffusion_policy/model/diffusion/ema_model.py`(88 行,几乎逐行照搬)。
差异:把 `new_model` 参数名改为 `new_model`,接受 `model: nn.Module` 深拷贝。
"""
import copy
import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel:
    """Exponential Moving Average of model weights.

    Usage:
        ema_model = copy.deepcopy(policy)
        ema = EMAModel(ema_model)
        for step in range(num_steps):
            loss = compute_loss(policy, batch)
            optim.step()
            ema.step(policy)   # ← in-place 更新 ema_model 权重

        # 评估时直接用 ema_model(它已经是 EMA 权重)
        pred = ema_model.predict_action(obs)
    """

    def __init__(self, model, update_after_step=0, inv_gamma=1.0,
                 power=2 / 3, min_value=0.0, max_value=0.9999):
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        """根据 new_model 的当前权重,更新 self.averaged_model 的 EMA 权重。"""
        self.decay = self.get_decay(self.optimization_step)

        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):
            for param, ema_param in zip(module.parameters(recurse=False),
                                          ema_module.parameters(recurse=False)):
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')

                if isinstance(module, _BatchNorm):
                    # skip batchnorms(直接复制,因为它们的统计量在 EMA 里也有累积意义)
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype),
                                    alpha=1 - self.decay)

        self.optimization_step += 1

    def state_dict(self):
        return {
            "averaged_model": self.averaged_model.state_dict(),
            "optimization_step": self.optimization_step,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict):
        self.averaged_model.load_state_dict(state_dict["averaged_model"])
        self.optimization_step = state_dict.get("optimization_step", 0)
        self.decay = state_dict.get("decay", 0.0)


def make_ema_from_policy(policy) -> EMAModel:
    """工具函数:对 policy 做 deepcopy 并包成 EMAModel。"""
    ema_model = copy.deepcopy(policy)
    return EMAModel(ema_model)
