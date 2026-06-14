"""B_model.encoders.VM.dp3_obs_encoder —— DP3 3D 点云编码器(真机/3D 仿真场景,阶段 3 启用)。

════════════════════════════════════════════════════════════════════════
安装命令(使用本类时按需安装,**不进** F_envs/base_train/environment.yml):
  # pip install termcolor  # cprint 打印用
  # (点云特征提取只用 torch,无需额外 pointnet2 编译)
════════════════════════════════════════════════════════════════════════

源:参考 PADP `model/vision/pointnet_extractor.py:DP3Encoder`(PointNet 编码点云 + state_mlp 编码 low_dim)。
VLA 阶段 1 robomimic 仿真**无点云**(只有 image + low_dim),本类阶段 3 真机/DP3 算法时启用。

════════════════════════════════════════════════════════════════════════
默认 config 注释(参考 PADP pointnet_extractor.py:DP3Encoder):
  # out_channel:           256
  # state_mlp_size:        (64, 64)
  # state_mlp_activation_fn: nn.ReLU
  # use_pc_color:          False           # True: 输入 (B, N, 6),False: (B, N, 3)
  # pointnet_type:         'pointnet'      # 或 'pointnet++'
  # state_keys:            ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
  # proj_dim:              64
════════════════════════════════════════════════════════════════════════
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from B_model.encoders.interface_vm import VisionEncoderInterface

logger = logging.getLogger(__name__)


def _create_mlp(input_dim, output_dim, net_arch, activation_fn=nn.ReLU, squash_output=False):
    """与 PADP `pointnet_extractor.py:create_mlp` 同款。"""
    modules = []
    if len(net_arch) > 0:
        modules += [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    for i in range(len(net_arch) - 1):
        modules += [nn.Linear(net_arch[i], net_arch[i + 1]), activation_fn()]
    if output_dim > 0:
        last_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """参考 PADP `PointNetEncoderXYZRGB`(在文件内联,不跨包依赖)。"""

    def __init__(self,
                 in_channels: int = 6,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = "none",
                 use_projection: bool = True):
        super().__init__()
        block_channel = [64, 128, 256, 512]
        self.mlp = nn.Sequential(*_create_mlp(in_channels, 0, block_channel, nn.ReLU))
        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def forward(self, x):
        # x: (B, N, in_channels)
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]  # (B, 512)
        x = self.final_projection(x)  # (B, out_channels)
        return x


class DP3ObsEncoder(VisionEncoderInterface):
    """DP3 3D 点云 + state 编码器(单点云/单 batch 接口)。

    输入约定:
      - point_cloud: (B, N, 3 or 6) Tensor(N 个点的 xyz [或 xyz+rgb])
      - state:       (B, D_state) Tensor(机械臂状态,concat 自 state_keys)
    输出:
      - (B, proj_dim)

    注:多相机/多模态由 Adapter 沿 batch 维 cat;本类只吃单点云。
    """

    def __init__(self,
                 out_channel: int = 256,
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 use_pc_color: bool = False,
                 pointnet_type: str = "pointnet",
                 state_keys=("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
                 proj_dim: int = 64):
        super().__init__()
        self.out_channel = out_channel
        self.use_pc_color = use_pc_color
        self.state_keys = list(state_keys)
        self.proj_dim = int(proj_dim)
        in_channels = 6 if use_pc_color else 3
        self.pointnet = PointNetEncoderXYZRGB(
            in_channels=in_channels,
            out_channels=out_channel,
            use_layernorm=False,
            final_norm="none",
        )
        # state MLP:本类**不**知道 state_dim(Adapter 喂进来时已 concat),用最大可能 dim 声明
        # 实际使用:Adapter 喂的 state shape 已知,这里用 placeholder,forward 时按 state.shape 适配
        self.state_mlp = nn.Sequential(
            *_create_mlp(input_dim=1, output_dim=out_channel, net_arch=list(state_mlp_size),
                          activation_fn=state_mlp_activation_fn)
        )
        self.proj = nn.Linear(out_channel * 2, proj_dim)
        logger.info("DP3ObsEncoder built: pc_in=%d, out_channel=%d, proj_dim=%d, pointnet=%s",
                    in_channels, out_channel, proj_dim, pointnet_type)

    def forward(self, point_cloud: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        """point_cloud: (B, N, 3 or 6);state: (B, D_state) or None

        Returns: (B, proj_dim)
        """
        # 点云特征
        pc_feat = self.pointnet(point_cloud)  # (B, out_channel)
        # state 特征(若有)
        if state is not None:
            # 重新过 MLP(state dim 不固定)
            state_feat = self._encode_state(state)  # (B, out_channel)
            feat = torch.cat([pc_feat, state_feat], dim=-1)  # (B, 2*out_channel)
        else:
            # 没有 state 时只用 pc feat,pad 一个零向量
            zero_state = torch.zeros(pc_feat.shape[0], self.out_channel,
                                     device=pc_feat.device, dtype=pc_feat.dtype)
            feat = torch.cat([pc_feat, zero_state], dim=-1)
        return self.proj(feat)

    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, D_state) → (B, out_channel)。按 state.shape 动态构造 MLP。"""
        d_state = state.shape[-1]
        # 重建 state_mlp(input_dim = d_state)
        mlp = nn.Sequential(
            *_create_mlp(input_dim=d_state, output_dim=self.out_channel,
                          net_arch=[self.out_channel // 2, self.out_channel // 2],
                          activation_fn=nn.ReLU)
        ).to(state.device)
        return mlp(state)

    def output_shape(self) -> tuple:
        return (self.proj_dim,)
