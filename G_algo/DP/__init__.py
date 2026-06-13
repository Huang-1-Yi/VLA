# -*- coding: utf-8 -*-
"""G_algo.DP —— 标准扩散策略(无位置加权,baseline)。"""
from .dp_pipeline import compute_loss, predict_action

__all__ = ["compute_loss", "predict_action"]
