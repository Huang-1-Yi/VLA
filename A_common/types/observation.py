"""契约:Observation —— 统一 obs 数据类,支持"什么有什么发什么"原则。"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch


@dataclass
class Observation:
    """统一 obs schema。

    rgb:   什么相机有什么填什么;key 约定 {'agentview_image', 'robot0_eye_in_hand_image', ...}
    state: 本体状态(关节+夹爪+eef pose),fp32 tensor
    prompt:自然语言指令(可选)
    timestamp:采集时刻
    """
    rgb: Dict[str, torch.Tensor] = field(default_factory=dict)
    state: Optional[torch.Tensor] = None
    prompt: Optional[str] = None
    timestamp: float = 0.0

    @classmethod
    def from_dict(cls, obs_dict: dict) -> "Observation":
        """工厂:从裸 dict 构造 Observation。"""
        return cls(
            rgb={k: v for k, v in obs_dict.items() if k.endswith(("_image", "_rgb"))},
            state=obs_dict.get("state"),
            prompt=obs_dict.get("prompt"),
            timestamp=obs_dict.get("timestamp", 0.0),
        )

    def to_dict(self) -> dict:
        """反向工厂:转回 dict(给 obs_encoder 入口用)。"""
        d = dict(self.rgb)
        if self.state is not None:
            d["state"] = self.state
        if self.prompt is not None:
            d["prompt"] = self.prompt
        if self.timestamp != 0.0:
            d["timestamp"] = self.timestamp
        return d
