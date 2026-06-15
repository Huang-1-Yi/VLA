# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""E_cti.train.padp_for_test_train —— VLA 端 1:1 复刻 PADP_v3 黄金命令训练效果。

==========================================================================
📋 架构分歧备忘录 (Divergence Memo)
==========================================================================

源命令:
    cd /media/disk7t/PADP_v3
    conda activate equidiff
    CUDA_VISIBLE_DEVICES=0 python train_sim.py --config-name=robomimic_padp_position_wise \\
      task_name=square_d0 n_demo=200 n_obs_steps=1 horizon=40 window_exp_gamma=0.25 \\
      policy.noise_schedule_mode=positionwise dataloader.batch_size=64 \\
      policy.pred_type=sample policy.noise_scheduler.prediction_type=sample \\
      exp_name=PADP_Golden_Standard

本文件 **复用 VLA 现有组件** 跑出"接近源端"训练效果,而不是逐行照搬源端代码。
下面列出 **VLA vs 源端的真实差异**,每条在文件内都有 `# >>> DIVERGENCE` 注释块。

┌───┬────────────────────┬─────────────────────┬─────────────────────┐
│ # │ 项目                │ 源端 (PADP_v3)       │ VLA (本文件)         │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│ 1 │ obs_encoder 架构    │ 单 bc_rnn,2 图+3 low │ 共享单图 bc_rnn,2  │
│   │                    │ dim → 137 维(22.4M) │ 相机 batch cat+    │
│   │                    │                     │ Linear→512 (11.8M)  │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│ 2 │ UNet global_cond   │ 137 维              │ 512 维              │
│   │   维数             │                     │ (架构分歧的连锁反应) │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│ 3 │ UNet 参数量         │ 63.0M               │ 68.4M               │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│ 4 │ 优化器 betas        │ (0.95, 0.999)       │ (0.95, 0.999) ✅    │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│ 5 │ LR 调度            │ cosine + 500 warmup │ cosine + 500 warmup │
│   │                    │ (PADP get_scheduler)│ (本文件 inline 实现) │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│ 6 │ EMA power          │ 0.75                │ 0.75 (显式传)       │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│ 7 │ 梯度裁剪            │ 无                  │ 无 ✅               │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│ 8 │ num_workers        │ 32                  │ 32 ✅               │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│ 9 │ drop_last          │ False               │ False ✅            │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│10 │ TopK ckpt 策略      │ 按 test_mean_score   │ 按 train_loss 降序   │
│   │                    │ 降序, k=5           │ k=5 (用户要求)      │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│11 │ Rollout 评估        │ 每 1000/n_demo 次   │ **完全跳过**         │
│   │                    │ 跑 env_runner        │ (用户明确要求)      │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│12 │ Validation 开环评估│ 默认注释掉          │ **完全跳过**         │
├───┼────────────────────┼─────────────────────┼─────────────────────┤
│13 │ Dataset            │ RobomimicReplayImage │ RobomimicZarrDataset│
│   │                    │ Dataset (RTV8)      │ Padp(RTV8-aligned)  │
│   │                    │ + per-epoch mapping │ + SequenceSampler   │
│   │                    │ (行=batch,列=sample)│   (默认,向后兼容)   │
│   │                    │                     │ + BalancedColumns-  │
│   │                    │                     │   Sampler(opt-in,   │
│   │                    │                     │   window_nums =     │
│   │                    │                     │   real_len+h-1,     │
│   │                    │                     │   行内强制不同 ep,  │
│   │                    │                     │   数学逻辑与 RTV8   │
│   │                    │                     │   _build_epoch_     │
│   │                    │                     │   mapping 1:1)     │
└───┴────────────────────┴─────────────────────┴─────────────────────┘

**结论**:差异 1/2/3 不可消除(它们是 VLA v5 架构的有意设计,改了就破坏 v5),
差异 4~12 已对齐源端,**差异 13 在 opt-in 模式下与 RTV8 数学逻辑 1:1 一致**
(window_nums 公式、pos_info 5 维、7→10 转换、平衡列贪心分配全部对齐)。
默认走 SequenceSampler(向后兼容),通过 yaml `data.balanced_sampler: true` 切到
RTV8-aligned BalancedColumnsSampler(DataLoader 自动转 shuffle=False + drop_last=True)。
训练出的模型与源端 **不可逐参数对比**,loss 曲线会有数值偏差。
如果后续发现性能差异,需进一步排查,详见 `padp_for_test_diverge_report.md`(待补)。

==========================================================================
复用文件清单(零修改)
==========================================================================
- Gpolicy.PADP.padp_policy.SlidingWindowDiffusionPolicy
- C_sim.robomimic.zarr_dataset_padp.RobomimicZarrDatasetPadp
- B_model.encoders.VM.robomimic_obs_encoder.RobomimicObsEncoder
- B_model.adapters.padp_adapter.PADPAdapter
- B_model.networks.DM.unet1d_padp.Unet1DPadp
- A_common.types.normalizer.LinearNormalizer
- A_common.types.normalizer_utils.robomimic_abs_action_only_normalizer_from_stat
- A_common.data.base_collator.base_collate
- A_common.registry.policy_registry.build_policy
- A_common.ckpt.ema.EMAModel
- A_common.logger.get_logger

==========================================================================
新增文件清单
==========================================================================
- E_cti/train/padp_for_test_train.py (本文件)
- E_cti/configs/padp_for_test_golden.yaml

==========================================================================
"""
import argparse
import copy
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 路径设置:把 VLA_ROOT 加到 sys.path(同 run_train.py 的做法)
_VLA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_VLA_ROOT))

import A_common
import B_model  # 触发 Policy 在 Gpolicy 里的 @register_policy 注册
import Gpolicy
import C_sim
from A_common.logger import get_logger
from A_common.registry.policy_registry import build_policy
from A_common.data.base_collator import base_collate
from A_common.ckpt.ema import EMAModel

logger = get_logger("padp_for_test_train")


# ====================================================================
# 嵌入的默认配置(原 padp_for_test_golden.yaml,1:1 复刻 PADP_v3 黄金命令)
# 不再依赖外部 yaml 文件;可通过 --config 覆盖。
# ====================================================================
DEFAULT_CONFIG = {
    "policy": {
        "name": "padp_unet",
        "shape_meta": "${data.shape_meta}",
        "horizon": 40,
        "n_obs_steps": 1,
        "n_action_steps": 1,
        "pred_type": "sample",
        "noise_schedule_mode": "positionwise",
        "noise_chunk_size": 1,
        "window_loss_weights": "exponential",
        "window_exp_gamma": 0.25,
        "window_min_weight": 0.02,
        "crop_shape": [76, 76],
        "obs_encoder_group_norm": True,
        "eval_fixed_crop": True,
        "down_dims": [256, 512, 1024],
        "kernel_size": 5,
        "n_groups": 8,
        "cond_predict_scale": True,
        "task_name": "square",
        "scheduler": {
            "num_train_timesteps": 40,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "squaredcos_cap_v2",
            "clip_sample": True,
            "set_alpha_to_one": True,
            "steps_offset": 0,
            "prediction_type": "sample",
        },
    },
    "data": {
        "n_demo": 200,
        "dataset_path": "data/robomimic/datasets/square_d0/square_d0_abs.hdf5",
        "horizon": 40,
        "n_obs_steps": 1,
        "n_action_steps": 1,
        "abs_action": True,
        "use_legacy_normalizer": False,
        # RTV8-aligned balanced-columns sampler 开关
        # false: 走 SequenceSampler + DataLoader(shuffle=True)  [默认]
        # true:  走 BalancedColumnsSampler (RTV8 1:1 复刻)
        "balanced_sampler": True,
        "batch_size": 64,
        "sampler_seed": 42,
        "shape_meta": {
            "obs": {
                "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
                "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
                "robot0_eef_pos": {"shape": [3]},
                "robot0_eef_quat": {"shape": [4]},
                "robot0_gripper_qpos": {"shape": [2]},
            },
            "action": {"shape": [10]},  # hdf5 实际 7D,dataset 内部做 7→10
        },
    },
    "train": {
        "num_epochs": 251,         # 源端 ${50000 / n_demo + 1} = 251
        "batch_size": 64,
        "num_workers": 32,         # 源端(对齐 PADP)
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": False,        # 源端不用 drop_last
        "device": "cuda:0",
        "log_every": 50,
        "lr": 1.0e-4,
        "betas": [0.95, 0.999],    # 源端
        "eps": 1.0e-8,
        "weight_decay": 1.0e-6,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 500,    # 源端
        "use_ema": True,
        "ema": {
            "update_after_step": 0,
            "inv_gamma": 1.0,
            "power": 0.75,          # 源端
            "min_value": 0.0,
            "max_value": 0.9999,
        },
        "no_grad_clip": True,      # 源端
        "ckpt_dir": "data/outputs/padp_for_test_golden",
        "resume": True,
        "seed": 42,
        "skip_rollout": True,
        "ckpt_topk": {
            "monitor_key": "train_loss",
            "mode": "min",
            "k": 5,
            "format_str": "epoch{epoch:03d}_loss{train_loss:.4f}.ckpt",
        },
    },
    "rollout": {"enabled": False},
    "logging": {
        "project": "padp_vla_for_test",
        "name": "padp_golden_standard_h40_obs1_a1_gamma025",
    },
}


# ====================================================================
# 配置解析(简单支持 ${a.b} 嵌套引用)
# ====================================================================
def _resolve(obj, root):
    if isinstance(obj, dict):
        return {k: _resolve(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(x, root) for x in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        v = root
        for p in obj[2:-1].split("."):
            v = v[p]
        return _resolve(v, root)
    return obj


def get_default_config() -> dict:
    """返回深拷贝的默认配置(避免外部修改污染原字典)。"""
    import copy as _copy
    return _copy.deepcopy(DEFAULT_CONFIG)


# ====================================================================
# >>> DIVERGENCE FROM SOURCE PADP_v3:
#   源端用 diffusion_policy.common.lr_scheduler.get_scheduler("cosine", ...)
#   本文件 inline 实现 cosine + 500 warmup,等效公式:
#       warmup 阶段:   lr = base_lr * step / warmup_steps
#       cosine 阶段:   lr = min_lr + 0.5 * (base_lr - min_lr) *
#                              (1 + cos(pi * progress))
# <<< END DIVERGENCE
# ====================================================================
def make_cosine_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                             min_lr_ratio: float = 0.0):
    """cosine + 线性 warmup 的 LR scheduler。等效 PADP get_scheduler("cosine", warmup=N)。"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ====================================================================
# TopK checkpoint manager (按 train_loss 升序保存最优 k 个)
# 源端 TopKCheckpointManager 按 test_mean_score 降序,本文件反转过来。
# ====================================================================
class LossTopKManager:
    """轻量 TopK ckpt manager:保留 train_loss 最低的 k 个 ckpt。"""

    def __init__(self, save_dir: Path, k: int = 5, format_str: str = "epoch{epoch:03d}_loss{train_loss:.4f}.ckpt"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.k = int(k)
        self.format_str = format_str
        # 维护一个最小堆,[(loss, path)]
        self.heap = []  # list of (loss, path)

    def try_save(self, epoch: int, train_loss: float, payload: dict) -> bool:
        """若 train_loss 排进前 k 最小,则保存并返回 True。"""
        if len(self.heap) < self.k:
            should_save = True
        else:
            # heap[0] 是当前最大 loss
            worst_loss = max(h[0] for h in self.heap)
            should_save = train_loss < worst_loss

        if not should_save:
            return False

        ckpt_path = self.save_dir / self.format_str.format(epoch=epoch, train_loss=train_loss)
        torch.save(payload, ckpt_path)
        logger.info("[TopK] Saved ckpt to %s (loss=%.6f)", ckpt_path, train_loss)
        self.heap.append((train_loss, str(ckpt_path)))
        # 若超出 k 个,删掉最差
        if len(self.heap) > self.k:
            self.heap.sort(key=lambda x: x[0])  # 升序
            self.heap.pop()  # 删最大
        return True

    def best(self):
        if not self.heap:
            return None, None
        sorted_h = sorted(self.heap, key=lambda x: x[0])
        return sorted_h[0]  # (最小 loss, path)


# ====================================================================
# 主函数
# ====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="可选:从 yaml 文件覆盖默认配置(不传则用嵌入的默认配置)")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="覆盖 config 里的 num_epochs(用于 smoke test)")
    args = parser.parse_args()

    if args.config is not None:
        import yaml as _yaml
        with open(args.config) as f:
            cfg = _yaml.safe_load(f)
        cfg = _resolve(cfg, cfg)
        logger.info("Loaded config from %s", args.config)
    else:
        cfg = get_default_config()
        cfg = _resolve(cfg, cfg)  # 解析 ${...} 嵌套引用
        logger.info("Using embedded default config (padp_for_test_golden)")

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]

    if args.max_epochs is not None:
        train_cfg["num_epochs"] = args.max_epochs
        logger.info("[smoke] Overriding num_epochs=%d", args.max_epochs)

    logger.info("=" * 80)
    logger.info("padp_for_test_train: 1:1 复刻 PADP_v3 黄金命令")
    logger.info("=" * 80)
    logger.info("data: dataset_path=%s n_demo=%d",
                data_cfg["dataset_path"], data_cfg["n_demo"])
    logger.info("train: num_epochs=%d batch=%d num_workers=%d lr=%.2e warmup=%d",
                train_cfg["num_epochs"], train_cfg["batch_size"], train_cfg["num_workers"],
                train_cfg["lr"], train_cfg["lr_warmup_steps"])
    logger.info("EMA: power=%s max=%s", train_cfg["ema"]["power"], train_cfg["ema"]["max_value"])

    # === 1. Dataset ===
    # >>> DIVERGENCE: 现有 C_sim.robomimic.zarr_dataset_padp.RobomimicZarrDatasetPadp
    #   用了 zarr v3 API (create_array),本环境 zarr=2.12 不支持。
    #   改用本目录下新增的 padp_for_test_dataset.RobomimicZarrDatasetPadpForTest
    #   (zarr v2 API 兼容版,行为 RTV8-aligned 等价)。
    from C_sim.robomimic.padp_for_test_dataset import RobomimicZarrDatasetPadpForTest
    # ============ 是否启用 RTV8-aligned balanced-columns sampler ============
    # data.balanced_sampler=True 时:
    #   - sampler 走 BalancedColumnsSampler(__len__ = B * max_cols,per-epoch 重建)
    #   - DataLoader 须配 shuffle=False + drop_last=True(sampler 自己已含洗牌)
    #   - 每 epoch 调 dataset.set_epoch(epoch) 重建 _global_mapping
    # 默认 False 时走 SequenceSampler + DataLoader(shuffle=True),向后兼容
    use_balanced = bool(data_cfg.get("balanced_sampler", False))
    dataset = RobomimicZarrDatasetPadpForTest(
        shape_meta=data_cfg["shape_meta"],
        dataset_path=data_cfg["dataset_path"],
        n_demo=data_cfg.get("n_demo", 200),
        horizon=data_cfg.get("horizon", 40),
        n_obs_steps=data_cfg.get("n_obs_steps", 1),
        n_action_steps=data_cfg.get("n_action_steps", 8),
        abs_action=data_cfg.get("abs_action", True),
        use_legacy_normalizer=data_cfg.get("use_legacy_normalizer", False),
        balanced_sampler=use_balanced,
        batch_size=data_cfg.get("batch_size") if use_balanced else None,
        sampler_seed=int(data_cfg.get("sampler_seed", 42)),
    )
    logger.info("Dataset built: %s, n_windows=%d, n_obs_steps=%d, horizon=%d, balanced=%s",
                type(dataset).__name__, len(dataset),
                dataset.n_obs_steps, dataset.horizon, use_balanced)

    # === 2. Normalizer ===
    normalizer = dataset.get_normalizer()
    logger.info("Normalizer built: keys=%s", list(normalizer._modules.keys()))

    # === 3. Policy (Fat Policy,内部装配 adapter + DM + scheduler) ===
    policy = build_policy(cfg["policy"])
    policy.set_normalizer(normalizer)
    device = torch.device(train_cfg["device"])
    policy.to(device)
    logger.info("Policy built: %s", type(policy).__name__)
    logger.info("Policy shape_info: %s", policy.shape_info())

    # === 4. DataLoader ===
    # >>> DIVERGENCE: 源端 num_workers=32, persistent_workers=True, 无 drop_last
    #     (VLA run_train 用 4 + drop_last=True,本文件改回源端值)
    # sampler-aware 配置:
    #   - balanced_sampler=True:  sampler 自己已含 per-epoch 洗牌,
    #                             故 shuffle=False, drop_last=True(否则末 batch 不齐)
    #   - balanced_sampler=False: 走源端默认 shuffle=True, drop_last=False
    if use_balanced:
        dl_shuffle = False
        dl_drop_last = True
    else:
        dl_shuffle = True
        dl_drop_last = bool(train_cfg.get("drop_last", False))
    dl = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=dl_shuffle,
        pin_memory=train_cfg.get("pin_memory", True),
        persistent_workers=train_cfg.get("persistent_workers", True)
                        and train_cfg["num_workers"] > 0,
        drop_last=dl_drop_last,
        collate_fn=base_collate,
    )
    logger.info("DataLoader: batch_size=%d iters_per_epoch=%d shuffle=%s drop_last=%s (sampler=%s)",
                train_cfg["batch_size"], len(dl), dl_shuffle, dl_drop_last,
                "balanced" if use_balanced else "sequence")

    # === 5. Optimizer (与源端对齐:AdamW betas=0.95/0.999) ===
    optim = torch.optim.AdamW(
        policy.parameters(),
        lr=train_cfg["lr"],
        betas=tuple(train_cfg.get("betas", [0.95, 0.999])),
        eps=train_cfg.get("eps", 1e-8),
        weight_decay=train_cfg.get("weight_decay", 1e-6),
    )
    total_steps = train_cfg["num_epochs"] * len(dl)
    lr_sched = make_cosine_with_warmup(
        optim,
        num_warmup_steps=train_cfg["lr_warmup_steps"],
        num_training_steps=total_steps,
    )
    logger.info("Optimizer: AdamW(betas=%s, eps=%.1e, wd=%.1e)",
                tuple(train_cfg.get("betas", [0.95, 0.999])),
                train_cfg.get("eps", 1e-8),
                train_cfg.get("weight_decay", 1e-6))
    logger.info("LR schedule: cosine + %d warmup, total_steps=%d",
                train_cfg["lr_warmup_steps"], total_steps)

    # === 6. EMA (与源端对齐:power=0.75) ===
    ema = None
    if train_cfg.get("use_ema", True):
        ema_model = copy.deepcopy(policy)
        ema_model.set_normalizer(policy._normalizer)
        ema = EMAModel(
            ema_model,
            update_after_step=train_cfg["ema"]["update_after_step"],
            inv_gamma=train_cfg["ema"]["inv_gamma"],
            power=train_cfg["ema"]["power"],          # >>> DIVERGENCE: 0.75, VLA EMAModel 默认 2/3
            min_value=train_cfg["ema"]["min_value"],
            max_value=train_cfg["ema"]["max_value"],
        )
        logger.info("EMA enabled: power=%s, max=%s",
                    train_cfg["ema"]["power"], train_cfg["ema"]["max_value"])

    # === 7. ckpt 路径 ===
    ckpt_dir = Path(train_cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "latest.ckpt"
    norm_ckpt = ckpt_dir / "normalizer.pt"

    # 独立保存 normalizer(不进 policy state_dict)
    torch.save(normalizer.state_dict(), norm_ckpt)

    # TopK manager
    topk = LossTopKManager(
        save_dir=ckpt_dir / "topk",
        k=train_cfg["ckpt_topk"]["k"],
        format_str=train_cfg["ckpt_topk"]["format_str"],
    )

    # === 8. Resume (源端行为) ===
    start_epoch = 0
    if train_cfg.get("resume", False) and latest_ckpt.exists():
        logger.info("Resuming from %s", latest_ckpt)
        ck = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["model_state"], strict=False)
        if "optim_state" in ck:
            optim.load_state_dict(ck["optim_state"])
        if ema is not None and "ema_state" in ck:
            ema.load_state_dict(ck["ema_state"])
        start_epoch = ck.get("epoch", 0) + 1

    # === 9. 训练循环 ===
    logger.info("=" * 80)
    logger.info("Starting training: %d epochs × %d iters = %d total steps",
                train_cfg["num_epochs"], len(dl), total_steps)
    logger.info("Skip rollout: %s", train_cfg.get("skip_rollout", True))
    logger.info("=" * 80)

    global_step = 0
    history = []  # 记录每 epoch 的 train_loss,用于 TopK
    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        # ====== balanced_sampler: 每 epoch 重建 _global_mapping(per-epoch 洗牌) ======
        if use_balanced:
            dataset.set_epoch(epoch)
        policy.train()
        epoch_loss_sum = 0.0
        epoch_n = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(dl):
            # 把数据搬到 GPU
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v)
                    else {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()})
                for k, v in batch.items()
            }
            # ⭐ E_cti 唯一动作:把 batch 丢进 policy,拿回 loss
            loss = policy.compute_loss(batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            # >>> DIVERGENCE: 源端不做 grad clip (VLA run_train 做了 1.0,本文件按源端关掉)
            if not train_cfg.get("no_grad_clip", True):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            lr_sched.step()
            if ema is not None:
                ema.step(policy)

            global_step += 1
            B = batch["action"].shape[0]
            epoch_loss_sum += float(loss.item()) * B
            epoch_n += B
            if global_step % train_cfg["log_every"] == 0:
                lr_now = optim.param_groups[0]["lr"]
                logger.info("ep=%d step=%d/%d loss=%.4f lr=%.2e",
                            epoch, batch_idx + 1, len(dl), float(loss.item()), lr_now)

        avg = epoch_loss_sum / max(1, epoch_n)
        elapsed = time.time() - t0
        logger.info("Epoch %d/%d: avg_loss=%.4f time=%.1fs lr=%.2e",
                    epoch, train_cfg["num_epochs"], avg, elapsed,
                    optim.param_groups[0]["lr"])
        history.append((epoch, avg))

        # === 9.5.  ROLLOUT 已禁用 (v1.0) ===
        # 黄金标准配置中 train.skip_rollout=True,默认不跑 rollout。

        # === 10. 保存 latest ckpt + normalizer ===
        ckpt_payload = {
            "model_state": policy.state_dict(),
            "optim_state": optim.state_dict(),
            "epoch": epoch,
            "train_loss": avg,
            # 🚧 同时把 normalizer 嵌进 ckpt,方便 server/client 直接加载
            # (不必再去 load normalizer.pt 路径,见 rollout 流程)
            "normalizer_state": policy._normalizer.state_dict() if policy._normalizer is not None else None,
        }
        if ema is not None:
            ckpt_payload["ema_state"] = ema.state_dict()
        torch.save(ckpt_payload, latest_ckpt)
        torch.save(policy._normalizer.state_dict(), norm_ckpt)

        # === 11. TopK ckpt (按 train_loss 升序,选最小 loss) ===
        topk.try_save(epoch=epoch, train_loss=avg, payload=ckpt_payload)

    # === 12. 训练结束报告 ===
    best_loss, best_path = topk.best()
    logger.info("=" * 80)
    logger.info("Training done. %d epochs completed.", train_cfg["num_epochs"])
    logger.info("Latest ckpt: %s", latest_ckpt)
    logger.info("Normalizer:  %s", norm_ckpt)
    if best_loss is not None:
        logger.info("BEST ckpt by train_loss: loss=%.6f  path=%s", best_loss, best_path)
    else:
        logger.info("No TopK ckpt saved (num_epochs < k?)")
    # 打印 loss 历史
    logger.info("Loss history:")
    for ep, loss in history:
        logger.info("  epoch %3d: loss=%.6f", ep, loss)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
