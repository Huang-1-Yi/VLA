# -*- coding: utf-8 -*-
"""B_model.adapters —— 算法特征融合器(v5 新增)。

由 Gpolicy/<algo>/<algo>_policy.py 在 __init__ 实例化。
"""
from B_model.adapters.base_adapter import BaseAdapter
from B_model.adapters.padp_adapter import PADPAdapter
from B_model.adapters.dp_adapter import DPAdapter

__all__ = ["BaseAdapter", "PADPAdapter", "DPAdapter"]