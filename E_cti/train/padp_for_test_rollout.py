# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================


"""E_cti.train.padp_for_test_rollout —— server+client rollout 编排(中文日志)。

> 这是 train 脚本调用的 rollout 入口。它负责:
>   1. 把当前 policy 状态 dump 成可独立加载的 ckpt(server 用)
>   2. spawn server 子进程(padp_for_test_server.py)
>   3. spawn client 子进程(F_envs.robomimic.tcp_rollout_client),
>      后者跑 N 个 episode,返回 test_mean_score
>   4. 关掉 server
>   5. 返回 test_mean_score 给 train 脚本
> 日志已中文化
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

from A_common.logger import get_logger

logger = get_logger("padp_for_test_rollout")

_VLA_ROOT = Path(__file__).resolve().parents[2]


def _find_free_port() -> int:
    """找一个当前空闲的 TCP 端口(给 server 用)。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    """轮询 server 是否开始 listen,timeout 内 True/False。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.5)
    return False


def run_rollout_via_server_client(policy, ema, cfg, epoch) -> float:
    """通过 server+client 跑 N 个 episode,返回 test_mean_score。

    Args:
        policy: 当前在线 policy(若 use_ema=True 且 ema 可用,优先用 ema.averaged_model)
        ema:     EMAModel 实例(可能为 None)
        cfg:     完整 config dict
        epoch:   当前 epoch

    Returns:
        test_mean_score (float)  - 出错时返回 0.0
    """
    rollout_cfg = cfg.get("rollout", {}) or {}
    server_cfg = rollout_cfg.get("server", {}) or {}
    client_cfg = rollout_cfg.get("client", {}) or {}

    n_test = int(rollout_cfg.get("n_test", 3))
    max_steps = int(rollout_cfg.get("max_steps", 400))
    host = server_cfg.get("host", "127.0.0.1")
    port = int(server_cfg.get("port", 0)) or _find_free_port()
    wait_timeout = float(server_cfg.get("wait_timeout_sec", 60))

    # === 1. Dump 临时 ckpt (server 加载用) ===
    tmp_ckpt_dir = Path(rollout_cfg.get("log_path", "data/outputs/padp_for_test_golden/rollout_logs"))
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_ckpt_path = tmp_ckpt_dir / f"rollout_ckpt_epoch{epoch:03d}.ckpt"

    # 选 model: ema (若可用) 还是 online policy?
    eval_model = ema.averaged_model if (ema is not None) else policy
    ckpt_payload = {
        "model_state": eval_model.state_dict(),
        "normalizer_state": policy._normalizer.state_dict() if hasattr(policy, "_normalizer") and policy._normalizer is not None else None,
        "epoch": epoch,
        "config": cfg,  # 喂给 server 用于还原 shape_meta
    }
    torch.save(ckpt_payload, tmp_ckpt_path)
    logger.info("[rollout] 已 dump rollout ckpt: %s (%.1f MB)",
                tmp_ckpt_path, tmp_ckpt_path.stat().st_size / 1e6)

    # === 2. Spawn server (policy) ===
    server_proc = None
    if server_cfg.get("spawn", True):
        server_script = _VLA_ROOT / "E_cti/train/padp_for_test_server.py"
        server_log = tmp_ckpt_dir / f"server_epoch{epoch:03d}.log"
        # 消融开关:server 是否在 EP_CHANGE 时真 reset
        server_reset_flag = str(rollout_cfg.get("server_reset_on_ep_change", True)).lower()
        server_cmd = [
            sys.executable, str(server_script),
            "--ckpt", str(tmp_ckpt_path),
            "--host", host, "--port", str(port),
            "--server_reset_on_ep_change", server_reset_flag,
        ]
        # 可选:用户显式给 config_path 才传 --config(否则用嵌入的 DEFAULT_CONFIG)
        if rollout_cfg.get("config_path"):
            server_cmd += ["--config", str(rollout_cfg["config_path"])]
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=open(server_log, "w"), stderr=subprocess.STDOUT,
        )
        logger.info("[rollout] 已 spawn server pid=%d, 等待端口 %d...", server_proc.pid, port)
        if not _wait_for_port(host, port, wait_timeout):
            logger.error("[rollout] Server 在 %ss 内未启动,放弃 rollout", wait_timeout)
            server_proc.kill()
            return 0.0
        logger.info("[rollout] Server 已就绪。")

    try:
        # === 3. Run client (env) — subprocess OR in-process ===
        test_mean_score = 0.0
        if client_cfg.get("spawn", True):
            # 用 F_envs/robomimic 里的 TCP client(由子 AI 实现)
            client_script = _VLA_ROOT / "F_envs/robomimic/tcp_rollout_client.py"
            client_log = tmp_ckpt_dir / f"client_epoch{epoch:03d}.log"
            client_cmd = [
                sys.executable, str(client_script),
                "--host", host, "--port", str(port),
                "--task_name", str(rollout_cfg.get("task_name", "square")),
                "--dataset_path", str(cfg["data"]["dataset_path"]),
                "--n_train", str(rollout_cfg.get("n_train", 2)),
                "--n_test", str(n_test),
                "--max_steps", str(max_steps),
            ]
            client_proc = subprocess.run(
                client_cmd,
                stdout=open(client_log, "w"), stderr=subprocess.STDOUT,
                timeout=int(rollout_cfg.get("client_timeout_sec", 600)),
            )
            # 解析 client stdout 里的 test_mean_score
            score_line = ""
            for line in reversed(client_log.read_text().splitlines()):
                if "TEST_MEAN_SCORE=" in line:
                    score_line = line
                    break
            if score_line:
                try:
                    test_mean_score = float(score_line.split("TEST_MEAN_SCORE=")[1].split()[0])
                except (ValueError, IndexError):
                    test_mean_score = 0.0
            logger.info("[rollout] client exit_code=%d, 解析得到 test_mean_score=%.4f",
                        client_proc.returncode, test_mean_score)
        else:
            # 调试用:in-process 调 client.run_all(由子 AI 实现)
            from F_envs.robomimic.tcp_rollout_client import PadpRolloutClient
            import json as _json
            client = PadpRolloutClient(
                host=host, port=port,
                task_name=str(rollout_cfg.get("task_name", "square")),
                dataset_path=str(cfg["data"]["dataset_path"]),
                shape_meta=cfg["data"]["shape_meta"],
                n_train=int(rollout_cfg.get("n_train", 2)),
                n_test=n_test, max_steps=max_steps,
                abs_action=bool(rollout_cfg.get("abs_action", True)),
            )
            client.connect()
            try:
                result = client.run_all()
                test_mean_score = float(result.get("test/mean_score", 0.0))
            finally:
                client.close()
            logger.info("[rollout] in-process client: test_mean_score=%.4f", test_mean_score)
    finally:
        # === 4. 收尾:关 server ===
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
            logger.info("[rollout] Server 已关闭。")

    return test_mean_score
