# -*- coding: utf-8 -*-
"""B_model.encoders.TM —— TimestepEncoderInterface 实现族。

本阶段只放 SinusoidalTimestepEncoder(抄 PADP)。
阶段 2/3 加 LearnableTimestepEncoder 等变体。
"""
from B_model.encoders.TM.sinusoidal import SinusoidalTimestepEncoder, SinusoidalPosEmb

__all__ = ["SinusoidalTimestepEncoder", "SinusoidalPosEmb"]