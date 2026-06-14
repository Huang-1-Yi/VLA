"""B_model.adapters.padp_adapter —— PADP 算法 Adapter。

PADP 算法 Adapter 形态(对照 PADP SlidingWindowDiffusionPolicy.compute_loss/predict_action):
  - VM (RobomimicObsEncoder) 提取 obs 特征 → [B, D_obs]
  - reshape 为 global_cond = obs_features.reshape(B, -1) → [B, n_obs_steps * D_obs]
  - 不用单独的 AM / TM encoder(PADP 把 noise schedule 绑到 position,不需要 timestep embedding)

由 Gpolicy/PADP/padp_policy.py 在 __init__ 实例化。
"""
import torch
import torch.nn as nn

from B_model.adapters.base_adapter import BaseAdapter
from B_model.encoders.VM.robomimic_obs_encoder import RobomimicObsEncoder


class PADPAdapter(BaseAdapter):
    """PADP 算法的特征融合器:VM 提特征,reshape 为 global_cond。"""

    def __init__(self, shape_meta: dict, crop_shape=(76, 76),
                 obs_encoder_group_norm: bool = True,
                 eval_fixed_crop: bool = True,
                 task_name: str = "square"):
        super().__init__()
        # 实例化 VM(从 B_model 选具体实现)
        self.vm = RobomimicObsEncoder(
            shape_meta=shape_meta,
            crop_shape=crop_shape,
            obs_encoder_group_norm=obs_encoder_group_norm,
            eval_fixed_crop=eval_fixed_crop,
            task_name=task_name,
        )

    def forward(self, obs) -> torch.Tensor:
        """输入 obs(dict,可能含 'rgb' 子 dict / 'state'),输出 global_cond [B, D_obs]。

        PADP 调用方负责调用前 flatten 时间维:
            this_obs = dict_apply(obs, lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            global_cond = adapter(this_obs)
        """
        return self.vm(obs)