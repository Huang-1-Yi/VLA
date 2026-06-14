"""B_model.adapters.dp_adapter —— DP (baseline) Adapter。

DP 与 PADP 共享同一个 UNet,只是 loss 计算时不做 per-position 加权。
Adapter 结构完全相同(VM 提特征),直接继承 PADPAdapter。
"""
from B_model.adapters.padp_adapter import PADPAdapter


class DPAdapter(PADPAdapter):
    """DP baseline:Adapter 与 PADP 一样,只是 loss 计算无位置加权。"""
    pass