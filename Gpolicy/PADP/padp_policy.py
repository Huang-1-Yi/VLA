"""Gpolicy.PADP.padp_policy —— PADP 算法 Fat Policy(策略大脑)。

源:抄自 PADP `diffusion_policy/policy/robomimic/diffusion_unet_hybrid_padp.py:SlidingWindowDiffusionPolicy`
   的全部 346 行,改接口对齐 v5 架构:

v5 改造要点:
  - 继承 A_common.types.base_policy.BasePolicy(替代 PADP 的 BaseImagePolicy)
  - __init__ 实例化 self.adapter = PADPAdapter(cfg)(替代 PADP 内部直接构造 RobomimicObsEncoder)
  - __init__ 实例化 self.denoiser = Unet1DPadp(...)(替代 PADP 内部 model)
  - __init__ 一次性实例化 self.noise_scheduler = DDIMScheduler(黄金标准:num_train_timesteps=horizon)
  - __init__ 预注册 buffer: alpha_bar / sqrt_alpha_bar / sqrt_one_minus_alpha_bar / window_weights_h
  - forward(a_t, t, global_cond): 单步 DM 封装(供 compute_loss / predict_action 内部调用)
  - compute_loss(batch): positionwise 加噪 + 加权 MSE(返回 scalar)
  - predict_action(obs): 推理入口,带 sliding window buffer + first-call initialization
  - @register_policy("padp_unet") 注册到 A_common.registry

E_cti 只调 2 个高层入口:
    loss = policy.compute_loss(batch)
    action = policy.predict_action(obs)
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from A_common.logger import get_logger
from A_common.registry.policy_registry import register_policy
from A_common.types.base_policy import BasePolicy
from A_common.types.action_output import ActionOutput
from B_model.adapters.padp_adapter import PADPAdapter
from B_model.networks.DM.unet1d_padp import Unet1DPadp

logger = get_logger(__name__)


def _apply_position_noise(self_, original_samples, noise, mode="positionwise"):
    """Position-aware 加噪(PADP 黄金标准)。
    0 位置 α_bar=1(纯动作),H-1 位置 α_bar=0(纯噪声)。
    """
    if mode in ("horizon", "positionwise") and hasattr(self_, "sqrt_alpha_bar_h"):
        return self_.sqrt_alpha_bar_h * original_samples + self_.sqrt_one_minus_alpha_bar_h * noise
    # Fallback(linear / constant / random 未在本阶段用)
    target_dtype = original_samples.dtype
    device = original_samples.device
    horizon = original_samples.shape[1]
    if mode == "linear":
        alpha_bar = torch.linspace(1.0 - 1.0 / horizon, 0.0, horizon, device=device, dtype=target_dtype)
        return alpha_bar.sqrt().view(1, horizon, 1) * original_samples + (1 - alpha_bar).sqrt().view(1, horizon, 1) * noise
    raise ValueError(f"Unsupported mode: {mode}")


@register_policy("padp_unet")
class SlidingWindowDiffusionPolicy(BasePolicy):
    """PADP 位置感知扩散 Fat Policy。

    Args:
        cfg: 配置 dict,含 shape_meta / horizon / n_obs_steps / n_action_steps /
             pred_type / noise_schedule_mode / window_loss_weights / window_exp_gamma /
             down_dims / kernel_size / n_groups / cond_predict_scale /
             crop_shape / obs_encoder_group_norm / eval_fixed_crop /
             scheduler.num_train_timesteps / scheduler.beta_schedule / scheduler.beta_start / scheduler.beta_end
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # ===== 解析 cfg =====
        shape_meta = cfg["shape_meta"]
        self.horizon = int(cfg["horizon"])
        self.n_obs_steps = int(cfg["n_obs_steps"])
        self.n_action_steps = int(cfg.get("n_action_steps", 1))
        self.action_dim = int(shape_meta["action"]["shape"][0])
        self.pred_type = cfg.get("pred_type", "sample")
        self.noise_schedule_mode = cfg.get("noise_schedule_mode", "positionwise")
        self.noise_chunk_size = int(cfg.get("noise_chunk_size", 1))
        window_loss_weights = cfg.get("window_loss_weights", "exponential")
        window_exp_gamma = float(cfg.get("window_exp_gamma", 0.2))
        window_min_weight = float(cfg.get("window_min_weight", 0.02))
        crop_shape = tuple(cfg.get("crop_shape", (76, 76)))
        obs_encoder_group_norm = bool(cfg.get("obs_encoder_group_norm", True))
        eval_fixed_crop = bool(cfg.get("eval_fixed_crop", True))
        task_name = cfg.get("task_name", "square")

        # ===== 算力零件装配(v5:Fat Policy 实例化 B_model 组件)=====
        self.adapter = PADPAdapter(
            shape_meta=shape_meta,
            crop_shape=crop_shape,
            obs_encoder_group_norm=obs_encoder_group_norm,
            eval_fixed_crop=eval_fixed_crop,
            task_name=task_name,
        )

        # global_cond_dim = obs_feature_dim * n_obs_steps
        # RobomimicObsEncoder 输出经我们加的 Linear 投影到 512,所以 raw=512
        obs_feature_dim = int(self.adapter.vm.output_shape()[0])
        global_cond_dim = obs_feature_dim * self.n_obs_steps

        self.denoiser = Unet1DPadp(
            input_dim=self.action_dim,
            global_cond_dim=global_cond_dim,
            down_dims=tuple(cfg.get("down_dims", (256, 512, 1024))),
            kernel_size=int(cfg.get("kernel_size", 5)),
            n_groups=int(cfg.get("n_groups", 8)),
            cond_predict_scale=bool(cfg.get("cond_predict_scale", True)),
        )

        # ===== 调度器实例化(黄金标准:DDIMScheduler, num_train_timesteps=horizon)=====
        scheduler_cfg = cfg.get("scheduler", {})
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=int(scheduler_cfg.get("num_train_timesteps", self.horizon)),
            beta_start=float(scheduler_cfg.get("beta_start", 0.0001)),
            beta_end=float(scheduler_cfg.get("beta_end", 0.02)),
            beta_schedule=scheduler_cfg.get("beta_schedule", "squaredcos_cap_v2"),
            clip_sample=bool(scheduler_cfg.get("clip_sample", True)),
            set_alpha_to_one=bool(scheduler_cfg.get("set_alpha_to_one", True)),
            steps_offset=int(scheduler_cfg.get("steps_offset", 0)),
            prediction_type=scheduler_cfg.get("prediction_type", "sample"),
        )
        # 黄金标准:用 PADP 自家的余弦表(不被 scheduler 内部 timesteps 干扰)
        self._build_horizon_alpha_bar()

        # ===== 窗口损失权重(PADP 特有,per-position)=====
        if window_loss_weights == "linear":
            window_weights = 1.0 - torch.arange(self.horizon, dtype=torch.float32) / self.horizon
        elif window_loss_weights == "exponential":
            window_weights = torch.exp(-torch.arange(self.horizon, dtype=torch.float32) * window_exp_gamma)
        elif window_loss_weights == "constant":
            window_weights = torch.ones(self.horizon)
        else:
            raise ValueError(f"Unknown window loss weights: {window_loss_weights}")
        if window_min_weight > 0.0:
            window_weights = torch.clamp(window_weights, min=window_min_weight)
        self.register_buffer("window_weights_h", window_weights.view(1, self.horizon, 1))

        # ===== 推理状态 =====
        self._inference_buffer: Optional[torch.Tensor] = None
        self._inference_global_cond: Optional[torch.Tensor] = None

        logger.info(
            "PADP Policy built: obs_feature_dim=%d, global_cond_dim=%d, "
            "horizon=%d, n_action_steps=%d, action_dim=%d, pred_type=%s, "
            "noise_mode=%s, window_exp_gamma=%.3f, scheduler=T=%d(%s)",
            obs_feature_dim, global_cond_dim, self.horizon, self.n_action_steps,
            self.action_dim, self.pred_type, self.noise_schedule_mode,
            window_exp_gamma, self.noise_scheduler.config.num_train_timesteps,
            self.noise_scheduler.config.beta_schedule,
        )

    def _build_horizon_alpha_bar(self):
        """构建 PADP 专属余弦表(长度 H+1,索引 0=纯净,索引 H=纯噪声)。"""
        H = self.horizon
        s = 0.008
        t = torch.arange(0, H + 1, dtype=torch.float32)
        alpha_bar = torch.cos((t / H + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        alpha_bar[0] = 1.0
        alpha_bar[-1] = 0.0
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar)
        self.register_buffer("sqrt_one_minus_alpha_bar", sqrt_one_minus_alpha_bar)

        # 预对齐 [1, H, 1] 供 O(1) 广播(positionwise 极速路径)
        self.register_buffer("sqrt_alpha_bar_h", sqrt_alpha_bar[1:H + 1].view(1, H, 1))
        self.register_buffer("sqrt_one_minus_alpha_bar_h", sqrt_one_minus_alpha_bar[1:H + 1].view(1, H, 1))

        # 末位噪声系数(严格为 1.0)
        self.register_buffer(
            "sqrt_one_minus_alpha_bar_tail",
            sqrt_one_minus_alpha_bar[H:H + 1].view(1, 1, 1),
        )

    # ===== BasePolicy 必须方法 =====

    def forward(self, a_t: torch.Tensor, t: torch.Tensor,
                global_cond: torch.Tensor) -> torch.Tensor:
        """单步 DM 封装(PADP 不使用 timestep embedding,只喂 global_cond)。"""
        return self.denoiser(a_t, local_cond=None, global_cond=global_cond)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """训练高层入口。E_cti 唯一动作:loss = policy.compute_loss(batch)。"""
        assert "obs" in batch and "action" in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])  # [B, H, D]
        B = nactions.shape[0]

        # 编码 obs(只取前 n_obs_steps 帧 flatten)
        this_obs = {
            k: (v[:, :self.n_obs_steps, ...].reshape(-1, *v.shape[2:]) if v.dim() >= 4 else v[:, :self.n_obs_steps, ...].reshape(B, -1))
            for k, v in nobs.items()
        }
        nobs_features = self.adapter(this_obs)        # [B*n_obs, D_obs]
        # 兼容 dict 与 Tensor 输入
        if isinstance(nobs_features, dict):
            # concat 所有特征(robomimic 多模态情形)
            nobs_features = torch.cat([v for v in nobs_features.values()], dim=-1)
        global_cond = nobs_features.reshape(B, -1)   # [B, n_obs * D_obs]

        # Position-aware 加噪
        noise = torch.randn_like(nactions)
        noisy = self.sqrt_alpha_bar_h * nactions + self.sqrt_one_minus_alpha_bar_h * noise

        # 单步前向
        pred = self.forward(noisy, t=None, global_cond=global_cond)

        # target 选择(epsilon vs sample)
        if self.pred_type == "epsilon":
            target = noise
        elif self.pred_type == "sample":
            target = nactions
        else:
            raise ValueError(f"Unsupported pred_type: {self.pred_type}")

        # Per-position weighted MSE
        loss_mse = F.mse_loss(pred, target, reduction="none")  # [B, H, D]
        loss_weighted = (loss_mse * self.window_weights_h).sum(dim=1)  # [B, D]
        loss_b = loss_weighted.mean(dim=-1)  # [B]
        return loss_b.mean()

    def predict_action(self, obs: dict):
        """推理高层入口。带 sliding window buffer + 首次调用预热。"""
        B = next(iter(obs.values())).shape[0]
        device = self.device
        dtype = self.dtype
        H = self.horizon
        D = self.action_dim

        # 初始化或更新 buffer + global_cond
        if self._inference_buffer is None or self._inference_buffer.shape[0] != B:
            self._initialize_inference_buffer(obs)
        else:
            nobs = self.normalizer.normalize(obs)
            this_obs = {
                k: (v[:, :self.n_obs_steps, ...].reshape(-1, *v.shape[2:]) if v.dim() >= 4 else v[:, :self.n_obs_steps, ...].reshape(B, -1))
                for k, v in nobs.items()
            }
            nobs_features = self.adapter(this_obs)
            if isinstance(nobs_features, dict):
                nobs_features = torch.cat([v for v in nobs_features.values()], dim=-1)
            self._inference_global_cond = nobs_features.reshape(B, -1)

        x0 = self._inference_buffer  # [B, H, D]
        noise = torch.randn_like(x0)
        noisy = self.sqrt_alpha_bar_h * x0 + self.sqrt_one_minus_alpha_bar_h * noise

        model_output = self.forward(noisy, t=None, global_cond=self._inference_global_cond)

        if self.pred_type == "epsilon":
            pred_actions = noisy - model_output
        elif self.pred_type == "sample":
            pred_actions = model_output
        else:
            raise ValueError(f"Unsupported pred_type: {self.pred_type}")

        # 取第一步动作 + buffer 左移
        action_to_execute = pred_actions[:, 0:self.n_action_steps, :]
        self._inference_buffer[:, :H - 1, :] = pred_actions[:, 1:, :]
        # 末位注入新噪声
        new_noise = torch.randn(B, 1, D, device=device, dtype=dtype)
        self._inference_buffer[:, H - 1:, :] = self.sqrt_one_minus_alpha_bar_tail * new_noise

        # 反归一化
        action_out = self.normalizer["action"].unnormalize(action_to_execute)
        return ActionOutput(actions=action_out.reshape(B, self.n_action_steps, D), is_chunk=True, latency_ms=0.0)

    def reset(self):
        """推理 episode 切换时清空 sliding window buffer。"""
        self._inference_buffer = None
        self._inference_global_cond = None

    # ===== 内部辅助 =====

    def _initialize_inference_buffer(self, obs_dict):
        """首次推理初始化:buffer = position-noise(zeros),然后跑 num_warm 步预热。"""
        B = next(iter(obs_dict.values())).shape[0]
        device = self.device
        dtype = self.dtype
        H = self.horizon
        D = self.action_dim

        base_noise = torch.randn(B, H, D, device=device, dtype=dtype)
        self._inference_buffer = self.sqrt_alpha_bar_h * torch.zeros_like(base_noise) + self.sqrt_one_minus_alpha_bar_h * base_noise

        # 编码 global_cond
        nobs = self.normalizer.normalize(obs_dict)
        this_obs = {
            k: (v[:, :self.n_obs_steps, ...].reshape(-1, *v.shape[2:]) if v.dim() >= 4 else v[:, :self.n_obs_steps, ...].reshape(B, -1))
            for k, v in nobs.items()
        }
        nobs_features = self.adapter(this_obs)
        if isinstance(nobs_features, dict):
            nobs_features = torch.cat([v for v in nobs_features.values()], dim=-1)
        self._inference_global_cond = nobs_features.reshape(B, -1)

        # 预热(模拟真实推理窗口滑动)
        num_warm = self.horizon + self.n_obs_steps - 1
        for _ in range(num_warm):
            x0 = self._inference_buffer
            noise = torch.randn_like(x0)
            noisy = self.sqrt_alpha_bar_h * x0 + self.sqrt_one_minus_alpha_bar_h * noise
            model_output = self.forward(noisy, t=None, global_cond=self._inference_global_cond)

            if self.pred_type == "epsilon":
                pred_actions = noisy - model_output
            elif self.pred_type == "sample":
                pred_actions = model_output
            else:
                raise ValueError(self.pred_type)

            self._inference_buffer[:, :H - 1, :] = pred_actions[:, 1:, :]
            new_noise = torch.randn(B, 1, D, device=device, dtype=dtype)
            self._inference_buffer[:, H - 1:, :] = self.sqrt_one_minus_alpha_bar_tail * new_noise

    def shape_info(self) -> str:
        return (f"SlidingWindowDiffusionPolicy(adapter.vm={self.adapter.vm.output_shape()}, "
                f"horizon={self.horizon}, n_obs_steps={self.n_obs_steps}, "
                f"action_dim={self.action_dim}, pred_type={self.pred_type})")