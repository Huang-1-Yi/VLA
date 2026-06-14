"""VM(阶段 1):RobomimicObsEncoder —— 沿用 PADP 工作版本,加少量接口改造。

源:几乎完全照搬 PADP `diffusion_policy/model/vision/robomimic_obs_encoder.py`,
   删去 import padp.* 的依赖,改为 import robomimic.* 直引。
"""
import torch
import torch.nn as nn
import numpy as np

import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
from robomimic.config import config_factory
import robomimic.scripts.generate_paper_configs as gpc
from robomimic.scripts.generate_paper_configs import (
    modify_config_for_default_image_exp,
    modify_config_for_default_low_dim_exp,
    modify_config_for_dataset,
)

from A_common.logger import get_logger
from A_common.types.vision_encoder import VisionEncoderInterface

logger = get_logger(__name__)


def _get_robomimic_config(algo_name="bc_rnn", hdf5_type="image",
                          task_name="square", dataset_type="ph"):
    """抄自 PADP `diffusion_policy/common/robomimic_config_util.py`。
    阶段 1 内联进 obs_encoder,避免跨层 import。"""
    base_dataset_dir = "/tmp/null"
    filter_key = None

    modifier_for_obs = modify_config_for_default_image_exp
    if hdf5_type in ("low_dim", "low_dim_sparse", "low_dim_dense"):
        modifier_for_obs = modify_config_for_default_low_dim_exp

    algo_config_name = "bc" if algo_name == "bc_rnn" else algo_name
    config = config_factory(algo_name=algo_config_name)
    config = modifier_for_obs(config)
    # 新版 robomimic 签名:(config, task_name, dataset_type, hdf5_type, base_dataset_dir, filter_key=None)
    config = modify_config_for_dataset(
        config=config,
        task_name=task_name,
        dataset_type=dataset_type,
        hdf5_type=hdf5_type,
        base_dataset_dir=base_dataset_dir,
        filter_key=filter_key,
    )
    return config


def _replace_submodules(root_module, predicate, func):
    """PADP `common/pytorch_util.py` 里的工具,这里复制一份避免 import 跨层。
    防御:robomimic 的 encoder 里有 None 子模块(未使用的 modality),跳过它们。"""
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


class RobomimicObsEncoder(VisionEncoderInterface):
    """Hydra-instantiable wrapper around robomimic's observation encoder."""

    def __init__(self, shape_meta: dict, crop_shape=(76, 76),
                 obs_encoder_group_norm: bool = False,
                 eval_fixed_crop: bool = False,
                 task_name: str = "square",
                 proj_dim: int = 512):
        super().__init__()
        self.shape_meta = shape_meta
        self.proj_dim = int(proj_dim)

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = {}
        for key, attr in shape_meta["obs"].items():
            obs_key_shapes[key] = list(attr["shape"])
            if attr.get("type", "low_dim") == "rgb":
                obs_config["rgb"].append(key)
            else:
                obs_config["low_dim"].append(key)

        # === 取 bc_rnn 的默认 config,作为 obs encoder 的脚手架 ===
        # 新版 robomimic 移除了 algo_factory._algo_config_to_dict,改为本地 helper。
        config = _get_robomimic_config(
            algo_name="bc_rnn", hdf5_type="image",
            task_name=task_name, dataset_type="ph",
        )
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

        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu",
        )
        encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            encoder = _replace_submodules(
                encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=max(1, x.num_features // 16),
                    num_channels=x.num_features,
                ),
            )

        if eval_fixed_crop:
            # 推理时把随机裁剪换成固定中心裁剪(避免随机性影响 eval)
            # VLA 阶段 1 没继承 PADP 的 diffusion_policy_compat,自己写一个等价类。
            import robomimic.models.obs_core as rmoc  # 新版 robomimic 把 CropRandomizer 移到 obs_core
            from B_model.encoders.VM.fixed_crop_randomizer import FixedCropRandomizer
            encoder = _replace_submodules(
                encoder,
                predicate=lambda x: isinstance(x, rmoc.CropRandomizer),
                func=lambda x: FixedCropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )

        self.encoder = encoder

        # === 关键修复:robomimic obs_encoder 输出未池化的高维特征(739k)直接灌给 UNet 会爆显存;
        # 加一个 Linear 投影到 proj_dim(默认 512),与 PADP 原版架构对齐 ===
        raw_dim = int(np.prod(self.encoder.output_shape()))
        self.proj = nn.Linear(raw_dim, self.proj_dim)

        logger.info(
            "RobomimicObsEncoder built, raw=%d, projected=%d",
            raw_dim, self.proj_dim,
        )

    def forward(self, obs_dict):
        feat = self.encoder(obs_dict)        # [B, raw_dim]
        feat = feat.flatten(start_dim=1)      # 防 obs_encoder 返 [B, k, raw_dim/k]
        return self.proj(feat)                # [B, proj_dim]

    def output_shape(self):
        return (self.proj_dim,)
