"""E_cti.train.run_train.py —— 训练主入口(纯白痴执行台)。

铁律 1 强化版(v5):
  - 1-1 无 class
  - 1-2 无算法概念(不出现 add_noise / DDIMScheduler / F.mse_loss)
  - 1-3 不穿透 Policy 内部(不出现 policy.vm. / policy.dm. / policy.adapter.)

E_cti 唯一动作:
    loss = policy.compute_loss(batch)
"""
import sys
import time
import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import A_common
import B_model  # 触发 Policy 在 Gpolicy 里的 @register_policy 注册
import Gpolicy
import C_sim
from A_common.logger import get_logger
from A_common.registry.policy_registry import build_policy
from A_common.data.base_collator import base_collate

logger = get_logger("train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 简单解析 ${a.b} 引用
    def _resolve(obj):
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(x) for x in obj]
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            key = obj[2:-1]
            v = cfg
            for p in key.split("."):
                v = v[p]
            return _resolve(v)
        return obj
    cfg = _resolve(cfg)

    logger.info("Config loaded from %s", args.config)
    logger.info("Policy: %s", cfg["policy"]["name"])
    logger.info("Task: %s demo=%d horizon=%d batch_size=%d",
                cfg["data"]["dataset_path"], cfg["data"]["n_demo"],
                cfg["data"]["horizon"], cfg["train"]["batch_size"])

    # === 1. Dataset ===
    dataset = C_sim.make_dataset(cfg)
    logger.info("Dataset: %d windows, n_obs_steps=%d, horizon=%d",
                len(dataset), dataset.n_obs_steps, dataset.horizon)

    # === 2. Normalizer ===
    normalizer = dataset.get_normalizer()
    logger.info("Normalizer built: keys=%s", list(normalizer._modules.keys()))

    # === 3. Policy(Fat Policy,内部装配 adapter + DM + scheduler)===
    policy = build_policy(cfg["policy"])
    policy.set_normalizer(normalizer)
    device = torch.device(cfg["train"]["device"])
    policy.to(device)
    logger.info("Policy built: %s", type(policy).__name__)
    logger.info("Policy shape_info: %s", policy.shape_info())

    # === 4. DataLoader ===
    dl = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle=True,
        pin_memory=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
        drop_last=True,
        collate_fn=base_collate,
    )

    # === 5. Optimizer + LR ===
    optim = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    total_steps = cfg["train"]["num_epochs"] * len(dl)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

    # === 6. EMA ===
    from A_common.ckpt.ema import EMAModel
    import copy
    ema = None
    if cfg["train"].get("use_ema", True):
        ema_model = copy.deepcopy(policy)
        ema_model.set_normalizer(policy._normalizer)
        ema = EMAModel(ema_model, update_after_step=0, inv_gamma=1.0,
                       power=0.75, min_value=0.0, max_value=0.9999)
        logger.info("EMA enabled")

    # === 7. ckpt ===
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "latest.ckpt"
    norm_ckpt = ckpt_dir / "normalizer.pt"

    # 独立保存 normalizer(不进 policy state_dict)
    torch.save(normalizer.state_dict(), norm_ckpt)

    start_epoch = 0
    if cfg["train"].get("resume", False) and latest_ckpt.exists():
        logger.info("Resuming from %s", latest_ckpt)
        ck = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["model_state"], strict=False)
        if "optim_state" in ck:
            optim.load_state_dict(ck["optim_state"])
        if ema is not None and "ema_state" in ck:
            ema.load_state_dict(ck["ema_state"])
        start_epoch = ck.get("epoch", 0) + 1

    # === 8. 训练循环(E_cti 唯一动作:policy.compute_loss) ===
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
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v)
                         else {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()})
                     for k, v in batch.items()}

            # ⭐ E_cti 唯一动作:把 batch 丢进 policy,拿回 loss
            loss = policy.compute_loss(batch)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            lr_sched.step()

            if ema is not None:
                ema.step(policy)

            global_step += 1
            epoch_loss += float(loss.item()) * batch["action"].shape[0]
            epoch_n += batch["action"].shape[0]
            if global_step % cfg["train"]["log_every"] == 0:
                lr_now = optim.param_groups[0]["lr"]
                logger.info("ep=%d step=%d loss=%.4f lr=%.2e",
                            epoch, global_step, float(loss.item()), lr_now)

        avg = epoch_loss / max(1, epoch_n)
        logger.info("Epoch %d: avg_loss=%.4f, time=%.1fs", epoch, avg, time.time() - t0)

        # 定期保存
        if (epoch + 1) % cfg["train"]["eval_every"] == 0 or epoch == cfg["train"]["num_epochs"] - 1:
            ckpt_payload = {
                "model_state": policy.state_dict(),
                "optim_state": optim.state_dict(),
                "epoch": epoch,
            }
            if ema is not None:
                ckpt_payload["ema_state"] = ema.state_dict()
            torch.save(ckpt_payload, latest_ckpt)
            # 同步保存 normalizer(独立文件,resume 用)
            torch.save(policy._normalizer.state_dict(), norm_ckpt)
            logger.info("Saved latest ckpt to %s + %s", latest_ckpt, norm_ckpt)

    logger.info("Training done. Latest ckpt: %s", latest_ckpt)


if __name__ == "__main__":
    main()