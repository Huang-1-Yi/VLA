# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""B_model.encoders.VM —— 视觉编码器(单图接口,输入 (B, C, H, W),输出 (B, D))。"""
from B_model.encoders.VM.robomimic_obs_encoder import RobomimicObsEncoder

__all__ = ["RobomimicObsEncoder"]
