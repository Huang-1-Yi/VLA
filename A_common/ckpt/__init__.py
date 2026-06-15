# -*- coding: utf-8 -*-
# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""A_common.ckpt —— checkpoint 工具(阶段 1 简单版,只存 model state_dict + normalizer + config)。"""
import torch
from pathlib import Path
from typing import Optional

from A_common.types.normalizer import LinearNormalizer
from .ema import EMAModel, make_ema_from_policy


def save_checkpoint(path: str, model: torch.nn.Module,
                    normalizer: Optional[LinearNormalizer] = None,
                    ema: Optional["EMAModel"] = None,
                    cfg: Optional[dict] = None,
                    epoch: int = 0, **kwargs):
    """统一 checkpoint 格式。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "cfg": cfg or {},
    }
    if normalizer is not None:
        payload["normalizer"] = normalizer.state_dict()
    if ema is not None:
        payload["ema_state"] = ema.state_dict()
    if kwargs:
        payload["extra"] = kwargs
    torch.save(payload, p)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
