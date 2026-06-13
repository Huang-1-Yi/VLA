# -*- coding: utf-8 -*-
"""G_algo.PADP —— PADP 算法主入口。

源:抄自 PADP `diffusion_policy/workspace/robomimic/train_padp_workspace_v3.py` 的
   compute_loss / predict_action 逻辑(去 workspace class)。
"""
from .padp_pipeline import compute_loss, predict_action

__all__ = ["compute_loss", "predict_action"]
