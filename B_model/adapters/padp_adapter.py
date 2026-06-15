# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================

"""B_model.adapters.padp_adapter —— PADP 算法 Adapter(多相机 + 多时间步版)。

v5-1 简化后:**Adapter 负责多相机 + 多时间步管理**,VM 只管单图。

调用流(每个 rgb_key 独立 rgb image):
  obs_dict:  {rgb_key: (B, T, C, H, W), lowdim_key: (B, T, D) or (B, D), ...}
       ↓
  对每个 rgb_key:
    1. 沿 batch 维 cat  rgb_imgs (各 camera) → (B*T*N, C, H, W)
    2. 调 self.vm (单图 encoder) → (B*T*N, D_vm)
    3. 沿 feature 维 cat  N 个 camera 输出 (reshape 回 (B, T, N, D_vm) → (B, T, N*D_vm))
  对 lowdim:
    4. 调 self.am (单步 AM) 或直传 → (B, T, D_am) or (B, T, D_lowdim)
    5. reshape to (B, T, *)
  融合:
    6. self.fusion(cat([rgb_feat, lowdim_feat]))  → (B, T, D_proj)

关键不变量:
  - VM 接口:forward(x: Tensor[B, C, H, W]) → Tensor[B, D_vm]
  - AM 接口:forward(x: Tensor[B, T, D_in]) → Tensor[B, T, D_am]
  - Adapter 在 (B, T) 维上不缩减,只 cat → fusion,policy 拿 (B, T, D_proj)
  - output_shape 报告 (D_proj,) —— Policy 拿这个去构造 DM 的 global_cond_dim
"""
import logging
import torch
import torch.nn as nn

from B_model.adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class PADPAdapter(BaseAdapter):
    """PADP 算法的特征融合器:Adapter 负责多相机沿 batch 维 cat + 多时间步保留。

    Args:
        shape_meta:       dict,用于知道 rgb_keys / low_dim_keys / action_dim
        vm_encoder:        单图 VM 编码器(已构造)
        am_encoder:        可选,low-dim 状态编码器;None 则直传
        proj_dim:          Adapter 输出维度(供 DM 接收)
    """

    def __init__(self,
                 shape_meta: dict,
                 vm_encoder: nn.Module,
                 am_encoder: nn.Module = None,
                 proj_dim: int = 512):
        super().__init__()
        self.shape_meta = shape_meta
        self.proj_dim = int(proj_dim)

        # 解析 key 列表
        self.rgb_keys = sorted(k for k, v in shape_meta["obs"].items()
                              if v.get("type", "low_dim") == "rgb")
        self.low_dim_keys = sorted(k for k, v in shape_meta["obs"].items()
                                   if v.get("type", "low_dim") != "rgb")
        self.n_cameras = len(self.rgb_keys)

        # 实例化 VM / AM
        self.vm = vm_encoder
        self.am = am_encoder

        # 融合网络:永远用 Linear 投影到 proj_dim(固定输出 dim,policy 拿这个去配 DM)
        D_vm = vm_encoder.output_shape()[0]
        D_am = am_encoder.output_shape()[0] if am_encoder is not None else 0
        D_lowdim = self._infer_lowdim_dim() if am_encoder is None else 0
        fusion_in_dim = self.n_cameras * D_vm + D_am + D_lowdim
        self.fusion = nn.Linear(fusion_in_dim, proj_dim)
        self._output_dim = proj_dim

        self._output_shape_cache = (self._output_dim,)

        # === L1b:__init__ 末尾 logger.info ===
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("PADPAdapter built: vm=%s, am=%s, n_cameras=%d, "
                    "n_lowdim=%d, fusion_in_dim=%d, proj_dim=%d, total_params=%.3e",
                    type(vm_encoder).__name__,
                    type(am_encoder).__name__ if am_encoder else "None",
                    self.n_cameras, len(self.low_dim_keys),
                    fusion_in_dim, self.proj_dim, total_params)

    def _infer_lowdim_dim(self) -> int:
        """从 shape_meta 推断 low-dim 总维(若 am 是 None,Adapter 把 low-dim 直传输出)。"""
        total = 0
        for k, v in self.shape_meta["obs"].items():
            if v.get("type", "low_dim") == "rgb":
                continue
            shape = v.get("shape", [])
            if shape:
                total += int(shape[0]) if shape else 1
        return total

    def forward(self, obs: dict) -> torch.Tensor:
        """输入 obs dict(可能含多个 rgb key + 多个 low-dim key),输出 global_cond (B, T, proj_dim)。

        rgb keys 期望 (B, T, C, H, W);lowdim keys 期望 (B, T, D) or (B, D)。
        """
        # === L1a 形状契约(代码注释,不在 docstring 里)===
        # ===   input  obs dict: rgb keys (B, T_obs, C, H, W);lowdim keys (B, T_obs, D_s) or (B, D_s)===
        # ===   output (B, T_obs, Dproj)  ;Dproj = self.proj_dim(默认 512)===
        # === 内部步骤(每步 shape 标注)===
        # ===   1. 多相机 cat:       (B*T_obs*N, C, H, W)===
        # ===   2. VM 输出:          (B*T_obs*N, D_vm)===
        # ===   3. reshape + flatten: (B, T_obs, N*D_vm)===
        # ===   4. AM 输出(若有):    (B, T_obs, D_am)===
        # ===   5. cat 融合:         (B, T_obs, N*D_vm + D_am)===
        # ===   6. fusion Linear:     (B, T_obs, Dproj)===
        # === 重要:output_shape() 返回 (Dproj,) 单步维;Policy 拿这个配 DM 的 global_cond_dim===
        # 0. 推断 batch + time
        if self.rgb_keys:
            sample = obs[self.rgb_keys[0]]
            if sample.dim() == 5:
                B, T = sample.shape[:2]
            else:                              # 4D: (B, C, H, W),当作 T=1
                B, T = sample.shape[0], 1
        elif self.low_dim_keys:
            sample = obs[self.low_dim_keys[0]]
            if sample.dim() == 3:
                B, T = sample.shape[:2]
            else:
                B, T = sample.shape[0], 1
        else:
            B = next(iter(obs.values())).shape[0]
            T = 1

        # 1. 多相机:沿 batch 维 cat
        if self.n_cameras > 0:
            rgb_per_cam = []
            for k in self.rgb_keys:
                img = obs[k]
                if img.dim() == 5:                # (B, T, C, H, W) → (B*T, C, H, W)
                    img = img.reshape(B * T, *img.shape[2:])
                elif img.dim() == 4:              # 已经是 (B, C, H, W),T=1
                    pass
                else:
                    raise ValueError(f"rgb obs {k} should be 4D or 5D, got {img.dim()}D")
                rgb_per_cam.append(img)
            batched = torch.cat(rgb_per_cam, dim=0)            # (B*T*N, C, H, W)
            rgb_feat = self.vm(batched)                         # (B*T*N, D_vm)
            N = self.n_cameras
            rgb_feat = rgb_feat.view(B, T, N, -1).flatten(2)   # (B, T, N*D_vm)
        else:
            rgb_feat = None

        # 2. low-dim
        if self.low_dim_keys:
            lowdim_per_key = []
            for k in self.low_dim_keys:
                data = obs[k]
                if data.dim() == 2:               # (B, D) → (B, T=1, D)
                    data = data.unsqueeze(1)
                # 已是 (B, T, D)
                lowdim_per_key.append(data)
            if self.am is not None:
                # AM 期望 (B, T, D_in) → (B, T, D_am)
                am_in = torch.cat(lowdim_per_key, dim=-1) if len(lowdim_per_key) > 1 else lowdim_per_key[0]
                lowdim_feat = self.am(am_in)                  # (B, T, D_am)
            else:
                # 直传,concat 所有 low-dim keys 的 feature
                lowdim_feat = torch.cat(lowdim_per_key, dim=-1) if len(lowdim_per_key) > 1 else lowdim_per_key[0]
                # 扩展到 (B, T, D_lowdim) —— 可能 lowdim_feat 已经是 (B, T, D)
        else:
            lowdim_feat = None

        # 3. 拼接 + 融合(在 (B, T) 维上)
        parts = [p for p in [rgb_feat, lowdim_feat] if p is not None]
        if len(parts) == 0:
            raise RuntimeError("PADPAdapter.forward: no rgb or low-dim features")
        if len(parts) == 1:
            fused_3d = parts[0]                                # (B, T, D)
        else:
            fused_3d = torch.cat(parts, dim=-1)                 # (B, T, D_total)
        if self.fusion is None:
            global_cond = fused_3d                              # (B, T, D) — 直传(无 fusion)
        else:
            global_cond = self.fusion(fused_3d)                # (B, T, proj_dim)

        # ★ 关键:对齐 PADP 原版 `global_cond = nobs_features.reshape(batch_size, -1)`
        #  → 把 T 维直接 cat 进 feature 维,policy 拿到的就是 (B, T*D_proj)
        return global_cond.reshape(B, -1)                      # (B, T*proj_dim)

    def output_shape(self) -> tuple:
        """报告**单时间步**输出维度(去除 T 维)。Policy 用此决定 DM global_cond_dim。"""
        return self._output_shape_cache