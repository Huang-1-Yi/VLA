"""E_cti.train.run_eval.py —— 评估主入口(纯白痴执行台)。

铁律 1 强化版(v5):E_cti 不出现 scheduler / loss / 算法细节。
E_cti 唯一动作:action = policy.predict_action(obs)(由 runner.run 调用)

铁律 4:只 import C_sim 顶层,不准 from C_sim.robomimic.env_runner import ...
"""
import sys
import argparse
import json
from pathlib import Path
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import A_common
import B_model  # 触发 Policy 注册
import Gpolicy
import C_sim
from A_common.logger import get_logger
from A_common.registry.policy_registry import build_policy
from A_common.ckpt import load_checkpoint

logger = get_logger("eval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--out", type=str, default="data/outputs/eval_log.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    logger.info("Eval config: %s, ckpt: %s", args.config, args.ckpt)

    # === 加载 ckpt ===
    ck = load_checkpoint(args.ckpt)

    # === Policy ===
    policy = build_policy(cfg["policy"])
    policy.load_state_dict(ck["model_state"], strict=False)
    device = torch.device(cfg.get("eval", {}).get("device", cfg["train"]["device"]))
    policy.to(device)
    policy.eval()

    # === 注入 normalizer(优先从 ckpt 同目录的 normalizer.pt 读)===
    from pathlib import Path as _P
    from A_common.types.normalizer import LinearNormalizer
    norm_ckpt = _P(args.ckpt).parent / "normalizer.pt"
    if norm_ckpt.exists():
        norm = LinearNormalizer()
        norm.load_state_dict(torch.load(norm_ckpt, map_location="cpu", weights_only=False))
        policy.set_normalizer(norm)
        logger.info("Loaded normalizer from %s", norm_ckpt)
    elif "normalizer" in ck:
        norm = LinearNormalizer()
        norm.load_state_dict(ck["normalizer"])
        policy.set_normalizer(norm)
        logger.info("Loaded normalizer from ckpt payload")
    else:
        logger.warning("No normalizer found — eval will use raw (un-normalized) actions!")

    # === EMA(若有,加载 EMA 权重)===
    if "ema_state" in ck:
        from A_common.ckpt.ema import EMAModel
        import copy
        ema_model = copy.deepcopy(policy)
        ema = EMAModel(ema_model)
        ema.load_state_dict(ck["ema_state"])
        policy.load_state_dict(ema.averaged_model.state_dict())
        policy.eval()
        logger.info("Loaded EMA weights")

    # === C_sim factory 拿 runner ===
    runner = C_sim.make_eval_runner(cfg)

    # === 跑评估 ===
    n_test = args.n_test or cfg["eval"]["n_test"]
    metrics = runner.run(policy.predict_action, n_test=n_test)
    logger.info("=" * 60)
    logger.info("Eval %s: success_rate=%.4f (%d/%d)",
                cfg["sim"]["task_name"], metrics["success_rate"],
                int(sum(metrics["per_episode_success"])), metrics["n_episodes"])
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