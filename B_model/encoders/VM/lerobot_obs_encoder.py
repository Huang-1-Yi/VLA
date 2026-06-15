# ============================================================
# PADP-VLA v1.2
# 1.2 版本,新加 lerobot rgb_key 兼容(observation.image / observation.wrist_image)
# ============================================================

"""B_model.encoders.VM.lerobot_obs_encoder —— LerobotRobomimicObsEncoder (v1.2 lerobot 兼容版)。

fork 自 `robomimic_obs_encoder.py` (1.1 干净版),核心差异:
  - rgb_key 可以是 lerobot 标准名 (含 '.'),如 'observation.image' / 'observation.wrist_image'
  - robomimic 内部 nn.Module 注册 key 不允许 '.',所以内部用 sanitized 名:
      'observation.image' -> 'rgb_main'
      'observation.wrist_image' -> 'rgb_wrist'
  - shape_meta 仍保留 lerobot 原名(给 dataset / adapter / server 用)
  - 跟 1.1 行为 1:1 等价(同样的 bc_rnn encoder + Linear 投影)

调用方:
  - `Gpolicy/PADP/padp_policy.py` 在 1.1 用 RobomimicObsEncoder
  - lerobot 线由 `E_cti/train/padp_for_libero_train.py` 通过 monkey-patch 替换为
    LerobotRobomimicObsEncoder,policy 内部构造时自动用 lerobot 兼容版
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


class LerobotRobomimicObsEncoder(VisionEncoderInterface):
    """v1.2 lerobot 兼容版 RobomimicObsEncoder。

    跟 1.1 RobomimicObsEncoder 行为 1:1 等价,唯一差异是:
    - rgb_key 可能是 lerobot 标准名 (含 '.'),如 'observation.image' / 'observation.wrist_image'
    - 内部用 sanitized key (无 '.') 喂给 robomimic bc_rnn encoder,避免
      `KeyError: module name can't contain "."` 错误
    - shape_meta 仍保留 lerobot 原名(给 dataset / adapter / server 协议用)

    设计选择:
    - 内部 encoder 由 `_build_robomimic_bc_rnn_encoder()` 工厂构造
    - 输出 raw dim = 137(robomimic bc_rnn 默认),通过 Linear 投影到 proj_dim
    - 多相机由 Adapter 沿 batch 维 cat 多个图,本类不感知
    """

    # rgb_key 映射表:lerobot 标准名 -> 内部 sanitized 名
    _RGB_KEY_SANITIZE = {
        "observation.image": "rgb_main",
        "observation.wrist_image": "rgb_wrist",
    }

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

        # v1.2 lerobot 兼容:若默认 rgb_key 不在 shape_meta,自动找第一个 rgb key
        if rgb_key == "agentview_image" and rgb_key not in shape_meta.get("obs", {}):
            rgb_keys_in_meta = [
                k for k, v in shape_meta.get("obs", {}).items()
                if v.get("type", "low_dim") == "rgb"
            ]
            if rgb_keys_in_meta:
                rgb_key = sorted(rgb_keys_in_meta)[0]
                logger.info(
                    "[lerobot_obs_encoder] 默认 rgb_key 'agentview_image' 不在 shape_meta,"
                    " 改用第一个 rgb key: '%s'",
                    rgb_key,
                )
        self.rgb_key = rgb_key

        # 计算内部 sanitized key(robomimic nn.Module 注册 key 不能含 '.')
        if rgb_key in self._RGB_KEY_SANITIZE:
            self._internal_rgb_key = self._RGB_KEY_SANITIZE[rgb_key]
        else:
            self._internal_rgb_key = rgb_key.replace(".", "_") if "." in rgb_key else rgb_key

        # 构造内部 robomimic encoder(若调用方已传入则直接用,便于测试/复用)
        # 关键:这里传 internal_rgb_key (sanitized) 给 robomimic 工厂
        if internal_encoder is None:
            internal_encoder = _build_robomimic_bc_rnn_encoder(
                shape_meta=shape_meta,
                rgb_key=self._internal_rgb_key,
                crop_shape=crop_shape,
                obs_encoder_group_norm=obs_encoder_group_norm,
                eval_fixed_crop=eval_fixed_crop,
                task_name=task_name,
            )

        # 探测 raw output dim(若 probe 失败,fallback 到 PADP bc_rnn 默认 137)
        self.encoder = internal_encoder
        placeholder_key = f"_placeholder_state_{self._internal_rgb_key}"
        try:
            with torch.no_grad():
                mock_img = torch.zeros((1,) + tuple(shape_meta["obs"][self.rgb_key]["shape"]), dtype=torch.float32)
                mock_state = torch.zeros((1, 1), dtype=torch.float32)
                raw = self.encoder({self._internal_rgb_key: mock_img, placeholder_key: mock_state})
            if raw.dim() > 2:
                raw = raw.flatten(1)
            raw_dim = raw.shape[-1]
        except Exception:
            logger.warning("raw_dim probe failed; falling back to 137 (PADP bc_rnn default)")
            raw_dim = 137
        self.raw_dim = raw_dim

        # Linear 投影到 proj_dim
        self.proj = nn.Linear(raw_dim, proj_dim)
        logger.info("LerobotRobomimicObsEncoder(单图) built: rgb_key=%s, internal=%s, raw_dim=%d, proj_dim=%d",
                    rgb_key, self._internal_rgb_key, raw_dim, proj_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, C, H, W) → (B, proj_dim)

        内部 encoder(robomimic bc_rnn)期望 dict 输入,key 必须用 sanitized 名
        (不能含 '.'),所以这里用 self._internal_rgb_key 喂。

        v1.2 lerobot 兼容:
          - 1.1 obs_to_torch 透传 uint8 image(保持原 dtype)
          - 但训练时 dataset 把 rgb 转 float32/255.0 (跟 1.1 一致)
          - encoder.proj 是 float weight,接收 uint8 会 RuntimeError
          - 这里把 img 转 float + 归一化, 跟训练时一致
        """
        # === 形状:input (B, C, H, W) → output (B, D_vm) ===
        # === D_vm = self.proj_dim(默认 512,见 __init__)===
        # 跟训练 dataset 行为一致:uint8 → float32 + / 255
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        elif img.dtype != torch.float32:
            img = img.float()
        B = img.shape[0]
        placeholder_key = f"_placeholder_state_{self._internal_rgb_key}"
        wrapped = {
            self._internal_rgb_key: img,
            placeholder_key: torch.zeros((B, 1), dtype=img.dtype, device=img.device),
        }
        raw = self.encoder(wrapped)        # (B, raw_dim) 或 (B, 1, 1, raw_dim)
        if raw.dim() > 2:
            raw = raw.flatten(1)
        return self.proj(raw)

    def output_shape(self) -> tuple:
        return (self.proj_dim,)