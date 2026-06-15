# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""契约 1:ActionOutput —— 统一动作输出,抹平 step vs chunk。"""
from dataclasses import dataclass
import torch


@dataclass
class ActionOutput:
    """统一模型输出。

    字段:
        actions:        [H, D_a] 物理动作(已反归一化)
                        - H=1 表示自回归单步
                        - H>1 表示 Diffusion / Flow 输出的 chunk
        is_chunk:       True → H>1(走 ChunkInterpolator)
                        False → H=1(走 StepInterpolator)
        latency_ms:     推理耗时
    """
    actions: torch.Tensor       # [H, D_a] 或 [1, D_a]
    is_chunk: bool
    latency_ms: float

    def __post_init__(self):
        assert self.actions.dim() == 2, f"actions must be 2D [H, D_a], got {self.actions.shape}"
        H = self.actions.shape[0]
        if self.is_chunk:
            assert H > 1, f"is_chunk=True but H={H} (should be >1)"
        else:
            assert H == 1, f"is_chunk=False but H={H} (should be 1)"
