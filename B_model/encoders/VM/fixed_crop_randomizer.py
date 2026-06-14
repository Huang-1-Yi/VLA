"""FixedCropRandomizer —— 推理时把 robomimic 的 CropRandomizer 换成固定中心裁剪。

PADP 原版放在 `diffusion_policy_compat/crop_randomizer.py`,VLA 没引那个模块,
所以在这里写一个等价的 Randomizer 子类(只要 forward_in / forward_out + shape API)。

行为:
- 训练和推理都做**中心裁剪**(无随机性,eval-friendly)
- 不做 num_crops reshape(num_crops 视为 1)
"""
import torch
import torch.nn as nn

from robomimic.models.base_nets import Module
from robomimic.models.obs_core import Randomizer


class FixedCropRandomizer(Randomizer):
    """中心裁剪版 randomizer,用于 eval / 固定 crop 训练。"""

    def __init__(self, input_shape, crop_height, crop_width, num_crops=1, pos_enc=False):
        super().__init__()
        assert len(input_shape) == 3  # (C, H, W)
        assert crop_height <= input_shape[1]
        assert crop_width <= input_shape[2]
        self.input_shape = tuple(int(x) for x in input_shape)
        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width)
        self.num_crops = int(num_crops)
        self.pos_enc = bool(pos_enc)
        # 中心点(以输入 H, W 为基准)
        self.center_y = (input_shape[1] - self.crop_height) // 2
        self.center_x = (input_shape[2] - self.crop_width) // 2

    def _forward_in(self, inputs):
        """训练时:也走中心裁剪(eval-fixed 模式)。"""
        return self._center_crop(inputs)

    def _forward_in_eval(self, inputs):
        return self._center_crop(inputs)

    def _forward_out(self, inputs):
        # 不做 num_crops reshape,直接返回
        return inputs

    def _forward_out_eval(self, inputs):
        return inputs

    def _center_crop(self, x):
        # x: [B, C, H, W]
        y0 = self.center_y
        y1 = y0 + self.crop_height
        x0 = self.center_x
        x1 = x0 + self.crop_width
        return x[:, :, y0:y1, x0:x1].contiguous()

    def output_shape_in(self, input_shape=None):
        if input_shape is None:
            return self.input_shape
        return (input_shape[0], self.crop_height, self.crop_width)

    def output_shape_out(self, input_shape=None):
        if input_shape is None:
            return self.input_shape
        return input_shape