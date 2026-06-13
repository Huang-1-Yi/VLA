"""E_cti.train.run_eval.py —— 评估主入口(纯脚本,无 class)。

流程:
    1. 读 yaml
    2. 加载 ckpt
    3. C_sim.make_eval_runner(cfg) 拿 runner
    4. runner.run(predict_fn)
    5. 写 outputs/eval_log.json
"""
import os
import sys
import argparse
import json
from pathlib import Path
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import A_common  # noqa
import B_model  # noqa
import G_algo  # noqa
import C_sim    # noqa
from A_common.logger import get_logger
from A_common.registry.policy_registry import build_policy
from A_common.ckpt import load_checkpoint

logger = get_logger("eval")


def resolve_refs(cfg: dict) -> dict:
    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_walk(x) for x in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            key = obj[2:-1]
            v = cfg
            for p in key.split("."):
                v = v[p]
            return v
        return obj
    return _walk(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--out", type=str, default="data/outputs/eval_log.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = resolve_refs(cfg)
    logger.info("Eval config: %s", args.config)
    logger.info("Ckpt: %s", args.ckpt)

    # === 加载 ckpt ===
    ck = load_checkpoint(args.ckpt)
    logger.info("Loaded ckpt: epoch=%s", ck.get("epoch"))

    # === 实例化 policy + 注入 normalizer ===
    policy = build_policy(cfg["policy"])
    policy.load_state_dict(ck["model_state"])
    if "normalizer" in ck:
        from A_common.types.normalizer import LinearNormalizer
        norm = LinearNormalizer()
        norm.load_state_dict(ck["normalizer"])
        policy.set_normalizer(norm)
    device = torch.device(cfg["train"]["device"])
    policy.to(device)
    policy.eval()

    # === EMA(若有,则加载 EMA 权重到 policy 覆盖)===
    if "ema_state" in ck:
        from A_common.ckpt.ema import EMAModel
        import copy
        ema_model = copy.deepcopy(policy)
        ema = EMAModel(ema_model)
        ema.load_state_dict(ck["ema_state"])
        # 把 EMA 权重搬到 policy
        policy.load_state_dict(ema.averaged_model.state_dict())
        policy.eval()
        logger.info("Loaded EMA weights (optimization_step=%d, decay=%.4f)",
                    ema.optimization_step, ema.decay)

    # === C_sim factory 拿 runner(铁律 4)===
    runner = C_sim.make_eval_runner(cfg)

    # === 跑评估 ===
    n_test = args.n_test or cfg["eval"]["n_test"]
    metrics = runner.run(policy.predict_action, n_test=n_test)
    logger.info("=" * 60)
    logger.info("Eval %s: success_rate=%.4f (%d/%d)",
                cfg["sim"]["task_name"],
                metrics["success_rate"],
                int(sum(metrics["per_episode_success"])),
                metrics["n_episodes"])
    logger.info("=" * 60)

    # === 写日志 ===
    out = {
        "config": args.config,
        "ckpt": args.ckpt,
        "task": cfg["sim"]["task_name"],
        "policy": cfg["policy"]["name"],
        "n_test": metrics["n_episodes"],
        "success_rate": metrics["success_rate"],
        "n_success": int(sum(metrics["per_episode_success"])),
        "per_episode_success": metrics["per_episode_success"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote eval log to %s", out_path)


if __name__ == "__main__":
    main()
