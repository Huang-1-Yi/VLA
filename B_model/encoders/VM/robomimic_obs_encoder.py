"""VM(阶段 1):RobomimicObsEncoder —— 沿用 PADP 工作版本,加少量接口改造。

源:几乎完全照搬 PADP `diffusion_policy/model/vision/robomimic_obs_encoder.py`,
   删去 import padp.* 的依赖,改为 import robomimic.* 直引。
"""
import torch
import torch.nn as nn

import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo

from A_common.logger import get_logger

logger = get_logger(__name__)


def _replace_submodules(root_module, predicate, func):
    """PADP `common/pytorch_util.py` 里的工具,这里复制一份避免 import 跨层。"""
    for name, module in root_module.named_children():
        if predicate(module):
            replaced = func(module)
            setattr(root_module, name, replaced)
        else:
            _replace_submodules(module, predicate, func)


class RobomimicObsEncoder(nn.Module):
    """Hydra-instantiable wrapper around robomimic's observation encoder."""

    def __init__(self, shape_meta: dict, crop_shape=(76, 76),
                 obs_encoder_group_norm: bool = False,
                 eval_fixed_crop: bool = False,
                 task_name: str = "square"):
        super().__init__()
        self.shape_meta = shape_meta

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

        # 这里走 robomimic 的 bc_rnn algo 取出 obs encoder
        config = algo_factory._algo_config_to_dict(
            algo_name="bc_rnn", hdf5_type="image",
            task_name=task_name, dataset_type="ph",
        )
        # 上面的 helper 不一定存在,改为手动 import config util
        from robomimic.config import config_factory
        config = config_factory(algo_name="bc_rnn", hdf5_type="image",
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
            import robomimic.models.base_nets as rmbn
            import diffusion_policy_compat.crop_randomizer as dmvc
            encoder = _replace_submodules(
                encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )

        self.encoder = encoder
        logger.info("RobomimicObsEncoder built, output_shape=%s", self.encoder.output_shape())

    def forward(self, obs_dict):
        return self.encoder(obs_dict)

    def output_shape(self):
        return self.encoder.output_shape()
