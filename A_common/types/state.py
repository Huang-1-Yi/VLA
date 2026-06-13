"""契约:State —— 统一本体状态。"""
from dataclasses import dataclass
import torch


@dataclass
class State:
    """本体状态(关节+夹爪+eef pose)。"""
    joint_pos: torch.Tensor       # [7]
    joint_vel: torch.Tensor = None  # [7]
    gripper: torch.Tensor = None    # [2]
    eef_pos: torch.Tensor = None    # [3]
    eef_quat: torch.Tensor = None   # [4]

    @property
    def vector(self) -> torch.Tensor:
        """拼成单向量(顺序:pos, vel, gripper, eef_pos, eef_quat)。"""
        parts = [self.joint_pos]
        if self.joint_vel is not None:
            parts.append(self.joint_vel)
        if self.gripper is not None:
            parts.append(self.gripper)
        if self.eef_pos is not None:
            parts.append(self.eef_pos)
        if self.eef_quat is not None:
            parts.append(self.eef_quat)
        return torch.cat(parts, dim=-1)
