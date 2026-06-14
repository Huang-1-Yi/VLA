# -*- coding: utf-8 -*-
"""B_model.encoders.AM —— ActionEncoderInterface 实现族。

本阶段只放 LinearStateEncoder(简单 nn.Linear);阶段 2/3 加 Conv1dEmbedder 等。
"""
from B_model.encoders.AM.linear_state_encoder import LinearStateEncoder

__all__ = ["LinearStateEncoder"]