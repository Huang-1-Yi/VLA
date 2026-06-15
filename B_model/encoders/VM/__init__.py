# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""B_model.encoders.VM —— 视觉编码器(单图接口,输入 (B, C, H, W),输出 (B, D))。"""
from B_model.encoders.VM.robomimic_obs_encoder import RobomimicObsEncoder

__all__ = ["RobomimicObsEncoder"]
