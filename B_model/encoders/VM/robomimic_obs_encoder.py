# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""B_model.encoders.VM.robomimic_obs_encoder —— RobomimicObsEncoder(VLA 单图版)。

v5-1 简化:
- 输入恒为单图 (B, C, H, W),输出恒为 (B, proj_dim)
- 多相机由 Adapter 沿 batch 维 cat 处理(本类不感知)
- _build_robomimic_bc_rnn_encoder() 是工厂,被 Adapter 用来构造 N 个 encoder 实例

继承 B_model.encoders.interface_vm.VisionEncoderInterface。
"""
import logging

import torch
import torch.nn as nn

from A_common.logger import get_logger
from B_model.encoders.interface_vm import VisionEncoderInterface

logger = get_logger(__name__)


def _replace_submodules(root_module, predicate, func):
    """递归把匹配 predicate 的子模块替换为 func(module)。防御 None 子模块。"""
    if root_module is None:
        return root_module
    for name, module in root_module.named_children():
        if module is None:
            continue
        if predicate(module):
            replaced = func(module)
            setattr(root_module, name, replaced)
        else:
            _replace_submodules(module, predicate, func)
    return root_module


def _get_robomimic_config(algo_name="bc_rnn", hdf5_type="image",
                          task_name="square", dataset_type="ph"):
    """抄自 PADP `diffusion_policy/common/robomimic_config_util.py`。
    VLA 阶段 1 内联,避免跨层 import。
    注意:新版 robomimic 移除了 `modify_config_for_low_dim_exp`,只保留
    `modify_config_for_default_image_exp`(因为我们只走 image 路径)。
    """
    from robomimic.config import config_factory
    from robomimic.scripts.generate_paper_configs import (
        modify_config_for_default_image_exp,
        modify_config_for_dataset,
    )
    base_dataset_dir = "/tmp/null"
    filter_key = None

    config = config_factory(algo_name=algo_name)
    config = modify_config_for_default_image_exp(config)
    config = modify_config_for_dataset(
        config=config,
        task_name=task_name,
        dataset_type=dataset_type,
        hdf5_type=hdf5_type,
        base_dataset_dir=base_dataset_dir,
        filter_key=filter_key,
    )
    return config


def _build_robomimic_bc_rnn_encoder(
    shape_meta: dict,
    rgb_key: str,
    crop_shape=(76, 76),
    obs_encoder_group_norm: bool = True,
    eval_fixed_crop: bool = True,
    task_name: str = "square",
) -> nn.Module:
    """工厂:为单个 rgb_key 构造一个 robomimic bc_rnn encoder 实例(已应用 BN→GN + crop)。

    Adapter 用本工厂给每个 rgb_key 构造一个独立 encoder,然后:
      1. 把 N 张图 cat 到 batch 维 (B*N, C, H, W)
      2. 同一个 encoder(若 share_rgb_model=True)或 N 个独立 encoder 各跑一次
      3. 拆分回 (B, N, D),再沿 feature 维 cat 成 (B, N*D)
    """
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.algo import algo_factory

    # 解析 shape_meta(只暴露单个 rgb_key + 一个固定 dim 的占位 lowdim,
    # 这样单图 forward 时 obs_utils 不会因缺模态而 assert 失败)
    action_shape = shape_meta["action"]["shape"]
    assert len(action_shape) == 1
    action_dim = action_shape[0]
    obs_key_shapes = {k: list(v["shape"]) for k, v in shape_meta["obs"].items() if v.get("type", "low_dim") == "rgb" and k == rgb_key}
    obs_key_shapes[f"_placeholder_state_{rgb_key}"] = [1]
    obs_config = {"low_dim": [f"_placeholder_state_{rgb_key}"],
                 "rgb": [rgb_key], "depth": [], "scan": []}

    config = _get_robomimic_config(algo_name="bc", hdf5_type="image",
                                    task_name=task_name, dataset_type="ph")
    with config.unlocked():
        config.observation.modalities.obs = obs_config
        if crop_shape is None:
            for _, modality in config.observation.encoder.items():
                if modality.obs_randomizer_class == "CropRandomizer":
                    modality["obs_randomizer_class"] = None
        else:
            ch, cw = crop_shape
            for _, modality in config.observation.encoder.items():
                if modality.obs_randomizer_class == "CropRandomizer":
                    modality.obs_randomizer_kwargs.crop_height = ch
                    modality.obs_randomizer_kwargs.crop_width = cw

    ObsUtils.initialize_obs_utils_with_config(config)

    # 构造 robomimic algo(它内部搭出 encoder)
    from robomimic.algo.algo import PolicyAlgo
    policy: PolicyAlgo = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=action_dim,
        device="cpu",
    )
    encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

    if obs_encoder_group_norm:
        _replace_submodules(
            encoder,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                num_channels=x.num_features,
            ),
        )

    return encoder


class RobomimicObsEncoder(VisionEncoderInterface):
    """Robomimic 单图编码器:接受 (B, C, H, W) → 输出 (B, proj_dim)。

    设计选择:
    - 内部 encoder 由 `_build_robomimic_bc_rnn_encoder()` 工厂构造
    - 输出 raw dim = 137(robomimic bc_rnn 默认),通过 Linear 投影到 proj_dim
    - 多相机由 Adapter 沿 batch 维 cat 多个图,本类不感知

    继承 B_model.encoders.interface_vm.VisionEncoderInterface。
    """

    def __init__(self,
                 shape_meta: dict,
                 proj_dim: int = 512,
                 rgb_key: str = "agentview_image",
                 crop_shape=(76, 76),
                 obs_encoder_group_norm: bool = True,
                 eval_fixed_crop: bool = True,
                 task_name: str = "square",
                 internal_encoder: nn.Module = None):
        super().__init__()
        self.shape_meta = shape_meta
        self.proj_dim = int(proj_dim)
        self.rgb_key = rgb_key

        # 构造内部 robomimic encoder(若调用方已传入则直接用,便于测试/复用)
        if internal_encoder is None:
            internal_encoder = _build_robomimic_bc_rnn_encoder(
                shape_meta=shape_meta,
                rgb_key=rgb_key,
                crop_shape=crop_shape,
                obs_encoder_group_norm=obs_encoder_group_norm,
                eval_fixed_crop=eval_fixed_crop,
                task_name=task_name,
            )

        # 探测 raw output dim(若 probe 失败,fallback 到 PADP bc_rnn 默认 137)
        self.encoder = internal_encoder
        placeholder_key = f"_placeholder_state_{rgb_key}"
        try:
            with torch.no_grad():
                mock_img = torch.zeros((1,) + tuple(shape_meta["obs"][rgb_key]["shape"]), dtype=torch.float32)
                mock_state = torch.zeros((1, 1), dtype=torch.float32)
                raw = self.encoder({rgb_key: mock_img, placeholder_key: mock_state})
            if raw.dim() > 2:
                raw = raw.flatten(1)
            raw_dim = raw.shape[-1]
        except Exception:
            logger.warning("raw_dim probe failed; falling back to 137 (PADP bc_rnn default)")
            raw_dim = 137
        self.raw_dim = raw_dim

        # Linear 投影到 proj_dim
        self.proj = nn.Linear(raw_dim, proj_dim)
        logger.info("RobomimicObsEncoder(单图) built: rgb_key=%s, raw_dim=%d, proj_dim=%d",
                    rgb_key, raw_dim, proj_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, C, H, W) → (B, proj_dim)

        内部 encoder(robomimic bc_rnn)期望 dict 输入,需要至少 rgb + 一个 low-dim placeholder。
        所以把单图包成 {rgb_key: img, placeholder_state: zeros} 再送进去。
        """
        # === 形状:input (B, C, H, W) → output (B, D_vm) ===
        # === D_vm = self.proj_dim(默认 512,见 __init__)===
        B = img.shape[0]
        placeholder_key = f"_placeholder_state_{self.rgb_key}"
        wrapped = {
            self.rgb_key: img,
            placeholder_key: torch.zeros((B, 1), dtype=img.dtype, device=img.device),
        }
        raw = self.encoder(wrapped)        # (B, raw_dim) 或 (B, 1, 1, raw_dim)
        if raw.dim() > 2:
            raw = raw.flatten(1)
        return self.proj(raw)

    def output_shape(self) -> tuple:
        return (self.proj_dim,)