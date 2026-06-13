"""E_cti.train.run_train.py —— 训练主入口(纯脚本,无 class)。

流程:
    1. 读 yaml
    2. C_sim.make_dataset(cfg)  → 训练数据
    3. dataset.get_normalizer() → 注入 policy
    4. A_common.registry.build_policy(cfg.policy) → 实例化
    5. policy.cuda()
    6. 选 G_algo.PADP.compute_loss 或 G_algo.DP.compute_loss
    7. AdamW + CosineLR + warmup
    8. 训练循环,每 N epoch 评估,top-k 保存
"""
import os
import sys
import time
import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

# === 让 import 找到顶层包 ===
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import A_common  # noqa
import B_model  # noqa
import G_algo  # noqa
import C_sim    # noqa
from A_common.logger import get_logger
from A_common.registry.policy_registry import build_policy
from A_common.ckpt import save_checkpoint

logger = get_logger("train")


def parse_overrides(overrides: list) -> dict:
    """CLI 覆盖 YAML 字段,支持 'key.path=value'。"""
    out = {}
    for kv in overrides:
        k, v = kv.split("=", 1)
        # 仅支持扁平 key
        out[k] = yaml.safe_load(v)
    return out


def merge_cfg(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = merge_cfg(out[k], v)
        else:
            out[k] = v
    return out


def resolve_refs(cfg: dict) -> dict:
    """把 ${a.b} 形式引用替换成实际值(简单实现,只支持扁平 ${key})。"""
    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_walk(x) for x in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            key = obj[2:-1]
            parts = key.split(".")
            v = cfg
            for p in parts:
                v = v[p]
            return v
        return obj
    return _walk(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    # === 1. 读 yaml + 覆盖 ===
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = resolve_refs(cfg)
    if args.overrides:
        ovr = parse_overrides(args.overrides)
        cfg = merge_cfg(cfg, ovr)

    logger.info("Config loaded from %s", args.config)
    logger.info("Policy: %s", cfg["policy"]["name"])
    logger.info("Task: square_d0 demo=%d horizon=%d batch_size=%d",
                cfg["data"]["n_demo"], cfg["data"]["horizon"], cfg["train"]["batch_size"])

    # === 2. C_sim.make_dataset ===
    dataset = C_sim.make_dataset(cfg)
    logger.info("Dataset: %d windows, n_obs_steps=%d, horizon=%d",
                len(dataset), dataset.n_obs_steps, dataset.horizon)

    # === 3. 注入 normalizer ===
    normalizer = dataset.get_normalizer()
    logger.info("Normalizer built: keys=%s", list(normalizer._modules.keys()))

    # === 4. A_common.registry.build_policy ===
    policy = build_policy(cfg["policy"])
    policy.set_normalizer(normalizer)
    device = torch.device(cfg["train"]["device"])
    policy.to(device)
    logger.info("Policy built: %s", type(policy).__name__)

    # === 5. DataLoader ===
    dl = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle=True,
        pin_memory=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
        drop_last=True,
    )

    # === 5.5 EMA(对照 PADP 修,bug fix 4)===
    import copy
    from A_common.ckpt.ema import EMAModel
    use_ema = cfg["train"].get("use_ema", True)
    ema = None
    if use_ema:
        ema_model = copy.deepcopy(policy)
        ema_model.set_normalizer(policy._normalizer)  # share same normalizer
        ema = EMAModel(ema_model, update_after_step=0, inv_gamma=1.0,
                       power=0.75, min_value=0.0, max_value=0.9999)
        logger.info("EMA enabled (update_after_step=0, inv_gamma=1.0, power=0.75, max_value=0.9999)")

    # === 6. 选 G_algo pipeline ===
    algo = cfg["policy"]["name"]
    if algo == "padp_unet":
        compute_loss = G_algo.PADP.compute_loss
    elif algo == "dp_unet":
        compute_loss = G_algo.DP.compute_loss
    else:
        raise ValueError(algo)

    # === 7. Optimizer + LR scheduler ===
    optim = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    total_steps = cfg["train"]["num_epochs"] * len(dl)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)
    # warmup
    warmup_steps = max(1, int(0.05 * total_steps))

    # === 8. ckpt 目录 ===
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "latest.ckpt"
    best_ckpt = ckpt_dir / "best.ckpt"
    best_success = 0.0

    # resume
    start_epoch = 0
    if cfg["train"].get("resume", False) and latest_ckpt.exists():
        logger.info("Resuming from %s", latest_ckpt)
        ck = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["model_state"])
        optim.load_state_dict(ck["optim_state"])
        if ema is not None and "ema_state" in ck:
            ema.load_state_dict(ck["ema_state"])
            logger.info("Resumed EMA at optimization_step=%d", ema.optimization_step)
        start_epoch = ck.get("epoch", 0) + 1
        best_success = ck.get("best_success", 0.0)
        logger.info("Resumed at epoch=%d, best_success=%.4f", start_epoch, best_success)

    # === 9. 训练循环 ===
    logger.info("=" * 80)
    logger.info("Starting training: %d epochs, %d iters/epoch, total %d steps",
                cfg["train"]["num_epochs"], len(dl), total_steps)
    logger.info("=" * 80)
    global_step = 0
    for epoch in range(start_epoch, cfg["train"]["num_epochs"]):
        policy.train()
        epoch_loss = 0.0
        epoch_n = 0
        t0 = time.time()
        for batch in dl:
            # batch 已经在 CPU;移到 device
            obs = {k: v.to(device, non_blocking=True) for k, v in batch["obs"].items()}
            act = batch["action"].to(device, non_blocking=True)
            # build inputs dict for G_algo
            inputs = {"obs": obs, "action": act}

            loss = compute_loss(policy, inputs)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()

            # warmup + cosine
            if global_step < warmup_steps:
                lr_now = cfg["train"]["lr"] * (global_step + 1) / warmup_steps
                for g in optim.param_groups:
                    g["lr"] = lr_now
            else:
                lr_sched.step()

            global_step += 1
            epoch_loss += float(loss.item()) * act.shape[0]
            epoch_n += act.shape[0]
            if global_step % cfg["train"]["log_every"] == 0:
                lr_now = optim.param_groups[0]["lr"]
                logger.info("ep=%d step=%d loss=%.4f lr=%.2e",
                            epoch, global_step, float(loss.item()), lr_now)

            # === EMA 更新(每步)===
            if ema is not None:
                ema.step(policy)

        avg = epoch_loss / max(1, epoch_n)
        logger.info("Epoch %d: avg_loss=%.4f, time=%.1fs", epoch, avg, time.time() - t0)

        # === 10. 定期保存 ===
        if (epoch + 1) % cfg["train"]["eval_every"] == 0 or epoch == cfg["train"]["num_epochs"] - 1:
            ckpt_payload = {
                "model_state": policy.state_dict(),
                "optim_state": optim.state_dict(),
                "epoch": epoch,
                "best_success": best_success,
            }
            if ema is not None:
                ckpt_payload["ema_state"] = ema.state_dict()
            torch.save(ckpt_payload, latest_ckpt)
            logger.info("Saved latest ckpt to %s", latest_ckpt)

    logger.info("=" * 80)
    logger.info("Training done. Best success=%.4f", best_success)
    logger.info("Latest ckpt: %s", latest_ckpt)
    logger.info("Best   ckpt: %s", best_ckpt)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
