# -*- coding: utf-8 -*-
"""B_model.encoders —— 编码器族(VM/AM/TM,本阶段)。

v5-1 §13 偏离版:每个 abstract interface 单独一个文件(最细粒度)。
- interface_vm.py     ← VisionEncoderInterface
- interface_am.py     ← ActionEncoderInterface
- interface_tm.py     ← TimestepEncoderInterface

导入顺序(避免循环 import):
  1) 先 re-export 3 个 abstract interface
  2) 再触发子包 VM/AM/TM(子包内的具体类继承 abstract)
"""
# 1) re-export abstract interface(最细粒度,每个 interface 单独文件)
from B_model.encoders.interface_vm import VisionEncoderInterface
from B_model.encoders.interface_am import ActionEncoderInterface
from B_model.encoders.interface_tm import TimestepEncoderInterface

# 2) 触发子包
from B_model.encoders import VM  # noqa: F401
from B_model.encoders import AM  # noqa: F401
from B_model.encoders import TM  # noqa: F401

__all__ = [
    "VisionEncoderInterface",
    "ActionEncoderInterface",
    "TimestepEncoderInterface",
    "VM", "AM", "TM",
]