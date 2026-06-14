"""A_common.types —— 数据契约(只剩 dataclass,abstract 接口已搬到各消费者层)。

v5-1 偏离版:5 个 abstract class(VisionEncoderInterface / ActionEncoderInterface /
TimestepEncoderInterface / DiffusionNetworkInterface / BasePolicy)已迁出本目录,
分别落到:
  - B_model/encoders/interface_vm_am_tm.py(VM/AM/TM 三件套)
  - B_model/networks/interface_dm.py(DM)
  - Gpolicy/base_policy_abstract.py(BasePolicy)

访问路径:
  - from A_common.types.action_output import ActionOutput  ← 本目录唯一用法
  - from B_model.encoders import VisionEncoderInterface    ← 跟消费者同包
  - from B_model.networks import DiffusionNetworkInterface ← 同上
  - from Gpolicy import BasePolicy                         ← 同上
"""
from .action_output import ActionOutput
from .observation import Observation
from .state import State
from .normalizer import LinearNormalizer, SingleFieldLinearNormalizer

__all__ = [
    "ActionOutput", "Observation", "State",
    "LinearNormalizer", "SingleFieldLinearNormalizer",
]