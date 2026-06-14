跟 PADP 原版完全对齐
PADP PADP_diffusion_unet_image_policy.py:


nobs_features = self.obs_encoder(this_obs)        # (B*T, D_obs)
global_cond = nobs_features.reshape(batch_size, -1)  # (B, T*D_obs)
VLA padp_policy.py (现在):


global_cond = self.adapter(this_obs)               # (B, T*proj_dim)  ← Adapter 内部已 flatten

```python
输入 obs_dict:
  agentview_image:        (B=2, T=1, C=3, H=84, W=84)
  robot0_eye_in_hand_image: (B=2, T=1, C=3, H=84, W=84)
  robot0_eef_pos:         (B=2, T=1, D=3)
  robot0_eef_quat:        (B=2, T=1, D=4)
  robot0_gripper_qpos:     (B=2, T=1, D=2)

↓ Adapter 内部:

# 1. 多相机沿 batch 维 cat
batched = cat([agentview, eye_in_hand], dim=0)    # (B*T*N, C, H, W) = (4, 3, 84, 84)
rgb_feat = vm(batched)                              # (4, 512)  [RobomimicObsEncoder 单图]
rgb_feat = rgb_feat.view(B, T, N, -1).flatten(2)   # (B, T, N*512) = (2, 1, 1024)

# 2. low-dim cat
lowdim_feat = cat([pos, quat, gripper], dim=-1)     # (B, T, 9) = (2, 1, 9)

# 3. 融合
fused_3d = cat([rgb_feat, lowdim_feat], dim=-1)     # (B, T, 1033)
global_cond = fusion(fused_3d)                      # (B, T, 512)   ← fusion_in_dim=1033 → proj_dim=512

# 4. ★ 关键:flatten T 进 feature
global_cond = global_cond.reshape(B, -1)             # (B, T*512) = (2, 512)  ← 对齐 PADP 原版

```

# B_model 输入输出结构清单

> 记录 `B_model/` 下所有 VM / AM / TM / Adapter 实现类的 **输入 shape → 输出 shape** 映射。
> 便于:policy 调用 `output_shape()` 自动适配 DM 全连接层 / Adapter 拼接 dim。

---

## 0. 通用约定

- **B**: batch size(可变)
- **T**: 时间步(`n_obs_steps`,默认 1)
- **N**: 相机数量(可变,如 square_d0 = 2)
- **D**: 输出特征维度(每个 encoder 的 `proj_dim` 决定,policy 通过 `output_shape()` 拿到)
- **H, W**: 图像高宽(默认 84×84)
- **C**: 图像通道(默认 3)
- **D_in**: state / action 输入维度
- 单图 vs 多图:**adapter** 决定是否拼接多相机输出,**policy 通过 `output_shape()` 自动适配**

---

## 1. VM(视觉编码器)—— `B_model/encoders/VM/`

### 1.1 单图编码器(每相机一个实例,Adapter 负责拼接)

| 类名 | 文件 | 输入 shape | 输出 shape | output_shape() | 参数量 | 备注 |
|---|---|---|---|---|---|---|
| `CLIPImageObsEncoder` | `clip_image_obs_encoder.py` | `(B, 3, 84, 84)` | `(B, proj_dim)` | `(proj_dim,)` | ~88M | 默认 proj_dim=512,需 `clip` 包 |
| `TimmImageObsEncoder` | `timm_image_obs_encoder.py` | `(B, 3, 84, 84)` | `(B, proj_dim)` | `(proj_dim,)` | ~11M(resnet18) | 默认 proj_dim=64,可选 resnet18/convnext/vit |

### 1.2 多图编码器(单个实例处理所有相机 / 所有 key)

| 类名 | 文件 | 输入 shape | 输出 shape | output_shape() | 参数量 | 备注 |
|---|---|---|---|---|---|---|
| `RobomimicObsEncoder` | `robomimic_obs_encoder.py` | `{key: (B, C, H, W)}` dict | `(B, 137)` raw → `(B, 512)` projected | `(512,)` | 22.5M(VM)+ 640(proj) | **PADP 工作版**,每个 key 一份独立 robomimic backbone,内部 BN→GN |
| `TimmObsEncoder` | `timm_obs_encoder.py` | `{key: (B, T, C, H, W)}` dict | `(B, T*proj_dim + sum_lowdim)` | 动态 | ~22M(resnet18 per-key) | 每个 rgb key deepcopy 一份 backbone,lowdim 直传 |
| `CLIPMultiImageObsEncoder` | `clip_multi_image_obs_encoder.py` | `{key: (B, T, C, H, W)}` dict | `(B, T*proj_dim + sum_lowdim)` | 动态 | ~88M(单 backbone) | **共享** CLIP backbone,所有 key 走同一个 |

### 1.3 单图 vs 多图 — 输出 shape 差异说明

| 相机数 | 选单图编码器 | 选多图编码器 |
|---|---|---|
| **1 个相机** | `(B, proj_dim)` | `(B, proj_dim)`(多图版本单 key 也输出同样 shape) |
| **N 个相机** | `(B, N × proj_dim)`(Adapter 拼接) | `(B, proj_dim)`(多图版本内部聚合) |

**关键不变量**:**两者的 `output_shape()` 都报告 `(proj_dim,)`**(因为 policy 看的是单 encoder 的输出维度,不知道有 N 个)。**Adapter / DM 拿到总 dim 时,要按 N × D 算**(详见 §3 Adapter)。

---

## 2. AM(动作 / 状态编码器)—— `B_model/encoders/AM/`

| 类名 | 文件 | 输入 shape | 输出 shape | output_shape() | 参数量 | 备注 |
|---|---|---|---|---|---|---|
| `LinearStateEncoder` | `linear_state_encoder.py` | `(B, T, D_in)` | `(B, T, proj_dim)` | `(proj_dim,)` | proj_dim × D_in + proj_dim | 简单 `nn.Linear(D_in → proj_dim)` |

> AM 当前只有一个实现(我们自己写)。PADP 没有专门 AM,直接 cat state,VLA 加这个是为了让 Adapter 拿到独立的 state 投影。

---

## 3. Adapter(特征融合器)—— `B_model/adapters/`

| 类名 | 文件 | 输入 obs 形态 | 输出 shape | output_shape() | 备注 |
|---|---|---|---|---|---|
| `PADPAdapter` | `padp_adapter.py` | `{key: (B, C, H, W)}` rgb + `{key: (B, D_lowdim)}` low_dim | `(B, 512)` global_cond | `(512,)` | RobomimicObsEncoder + Linear proj |
| `DPAdapter` | `dp_adapter.py` | 同上 | `(B, 512)` global_cond | `(512,)` | 继承 PADPAdapter |

> **Adapter 内部**:RobomimicObsEncoder 内部已经把多相机 concat,所以 Adapter 拿到 `(B, 137)`,Linear 投影到 `(B, 512)` 作为 DM 的 global_cond。
>
> 如果用单图编码器 + 2 相机,Adapter 需要把 `(B, D) + (B, D)` concat 为 `(B, 2D)`,再 Linear 投影到固定 dim(如 `(B, 512)`)。**这部分还没写**,阶段 1 用 RobomimicObsEncoder 跳过。

---

## 4. TM(扩散时间步编码器)—— `B_model/encoders/TM/`

| 类名 | 文件 | 输入 shape | 输出 shape | output_shape() | 参数量 | 备注 |
|---|---|---|---|---|---|---|
| `SinusoidalTimestepEncoder` | `sinusoidal.py` | `(B,)` timestep | `(B, dim)` | `(dim,)` | dim + dim·4·2 + dim·4·2 ≈ 9·dim² | `SinusoidalPosEmb + Linear + Mish + Linear`,抄 PADP |
| `SinusoidalPosEmb` | `sinusoidal.py`(内嵌) | `(B,)` timestep | `(B, dim)` | — | 0(纯数学) | `SinusoidalTimestepEncoder` 的子模块,单独 import |

> **TM 当前不被 PADP 算法使用**(PADP 用 position-based noise schedule,不需要 timestep)。**DP baseline 必须**(标准 diffusion)。

---

## 5. DM(扩散去噪网络)—— `B_model/networks/DM/`

| 类名 | 文件 | 输入 shape | 输出 shape | output_shape() | 备注 |
|---|---|---|---|---|---|
| `Unet1DPadp` | `unet1d_padp.py` | sample `(B, H, D_a)` + cond | `(B, H, D_a)` | `(D_a,)` | 主类,policy 直接调用 |

**DM 内部组件**(都是 nn.Module 子件,测试自动扫到):

| 类名 | 行 | 输入 | 输出 | 参数 |
|---|---|---|---|---|
| `Downsample1d` | 21 | `(B, dim, L)` | `(B, dim, L/2)` | dim·dim·3 |
| `Upsample1d` | 30 | `(B, dim, L)` | `(B, dim, 2L)` | dim·dim·4 |
| `Conv1dBlock` | 39 | `(B, inp, L)` | `(B, out, L)` | inp·out·3 |
| `ConditionalResidualBlock1D` | 53 | `(B, in, L)` + `(B, cond)` | `(B, out, L)` | (in·out·3 + out) × 2 + cond·out·2 |
| `FixedCropRandomizer` | `fixed_crop_randomizer.py` | `(B, 3, 84, 84)` | `(B, 3, 8, 8)` | 0 | robomimic 的 Randomizer 子类,eval 时中心裁剪 8×8 |

---

## 6. 输入输出总览(阶段 1 实际跑通的)

PADP 工作流(`square_d0` 黄金标准):

```
输入 obs:
  agentview_image:       (B, 3, 84, 84)
  robot0_eye_in_hand_image: (B, 3, 84, 84)
  robot0_eef_pos:        (B, 3)
  robot0_eef_quat:       (B, 4)
  robot0_gripper_qpos:    (B, 2)

↓ RobomimicObsEncoder (含 BN→GN + Linear proj)
输出:
  raw:                  (B, 137)
  projected global_cond: (B, 512)

↓ Unet1DPadp (DM)
输入 sample (B, 40, 7) + global_cond (B, 512)
输出:                  (B, 40, 7)
```

VLA 跑过的实际 loss 曲线(2026-06-14 02:22):
- step 50: 0.3475 → step 400: 0.0907(loss 持续下降)

---

## 7. 验证记录

**字节级对比**(2026-06-14 12:48,seed=0):
- PADP obs_encoder 输出:`[0.0, 0.2596909999847412, 0.0, 0.0, 0.05622720718383789, ...]`
- VLA RobomimicObsEncoder raw 输出:**完全一致**(8 位小数都相同)
- output_shape:**完全一致**(`(137,)` PADP / `(512,)` VLA projected)

**测试结果**:
- `test_encoders_output.py`: 6 ok, 0 fail(1 skip = SinusoidalPosEmb 是 SinusoidalTimestepEncoder 子模块)
- `test_networks_output.py`: 5 ok, 0 fail
- `test_adapters_output.py`: 2 ok, 0 fail