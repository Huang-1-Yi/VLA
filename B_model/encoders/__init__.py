# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""B_model.encoders —— 编码器族。"""
# Abstract interface
from B_model.encoders.interface_vm import VisionEncoderInterface

# 触发子包(VM 是 PADPAdapter 实际使用的)
from B_model.encoders import VM  # noqa: F401

__all__ = [
    "VisionEncoderInterface",
    "VM",
]
