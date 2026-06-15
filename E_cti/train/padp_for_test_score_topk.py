# ============================================================
# PADP-VLA v1.1
# 1.1 版本,补 rollout + best-ckpt by test_mean_score (max)
# ============================================================


"""E_cti.train.padp_for_test_score_topk —— 双 TopK ckpt 管理器。

> v1.1 pre-release 提供两个 TopK manager:
>   - LossTopKManager: 按 train_loss (min) 保留 Top-K (Phase 1: loss 阶段)
>   - ScoreTopKManager: 按 test_mean_score (max) 保留 Top-K (Phase 2: rollout 阶段)
>
> 双阶段策略:
>   - Phase 1 (loss 阶段): train_loss >= loss_threshold 时
>       - 用 LossTopKManager 选 top-K (min)
>       - 不跑 rollout (节省时间)
>   - Phase 2 (score 阶段): train_loss < loss_threshold 时
>       - 用 ScoreTopKManager 选 top-K (max)
>       - 每 N epoch 跑一次 rollout

类契约(两者一致,方便热替换):
  - __init__(save_dir, k, format_str)
  - try_save(epoch, score, payload) -> bool
  - best() -> (best_score, best_epoch, best_path)
"""
import logging
from pathlib import Path

import torch

from A_common.logger import get_logger

logger = get_logger("padp_for_test_score_topk")


# ============================================================
# LossTopKManager (Phase 1: 训练初期,按 train_loss 选 best)
# ============================================================
class LossTopKManager:
    """按 train_loss (min) 保留 Top-K ckpt。

    Phase 1 用,此时 rollout 还没触发(model 太嫩,score 没意义)。
    """

    def __init__(self,
                 save_dir: Path,
                 k: int = 3,
                 format_str: str = "epoch{epoch:03d}_loss{train_loss:.4f}.ckpt"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.k = int(k)
        self.format_str = format_str
        # 维护 entries: [(loss, epoch, path)],升序排列,头部是最小
        self.entries: list = []
        self._reload_existing()

    def _reload_existing(self):
        """从磁盘 reload 已有 loss_topk ckpt(支持续训)。"""
        for ckpt_path in self.save_dir.glob("epoch*_loss*.ckpt"):
            try:
                payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                loss = float(payload.get("train_loss", 999.0))
                epoch = int(payload.get("epoch", 0))
                self.entries.append((loss, epoch, str(ckpt_path)))
                logger.info("[LossTopK] 重新加载已有: %s loss=%.4f", ckpt_path.name, loss)
            except Exception as e:
                logger.warning("[LossTopK] 加载 %s 失败: %s", ckpt_path, e)
        # 升序(头部最小)
        self.entries.sort(key=lambda x: x[0])

    def try_save(self, epoch: int, train_loss: float, payload: dict) -> bool:
        """若 train_loss 排进前 k 最小,则保存并返回 True。"""
        # payload 加 train_loss 字段(供 reload)
        payload["train_loss"] = float(train_loss)
        payload["epoch"] = int(epoch)

        if len(self.entries) < self.k:
            should_save = True
        else:
            # 升序:头部最小(最优),但比较时取 max 的 (因为我们要保留最小的 k 个)
            worst_loss = self.entries[-1][0]  # 最后一个是最大的(最差)
            should_save = train_loss < worst_loss

        if not should_save:
            return False

        ckpt_path = self.save_dir / self.format_str.format(epoch=epoch, train_loss=train_loss)
        torch.save(payload, ckpt_path)
        logger.info("[LossTopK] 已保存 ckpt 到 %s (loss=%.6f)", ckpt_path, train_loss)
        self.entries.append((float(train_loss), int(epoch), str(ckpt_path)))

        # 升序排序(头部最小)
        self.entries.sort(key=lambda x: x[0])
        # 若超出 k 个,删尾部(最差)
        while len(self.entries) > self.k:
            _, _, old_path = self.entries.pop()  # pop 最后一个(最差)
            try:
                Path(old_path).unlink()
                logger.info("[LossTopK] 删除 loss 更高的 ckpt: %s", old_path)
            except OSError as e:
                logger.warning("[LossTopK] 删除 %s 失败: %s", old_path, e)
        return True

    def best(self):
        """返回 (best_loss, best_epoch, best_path)。空时返回 (None, None, None)。"""
        if not self.entries:
            return None, None, None
        sorted_h = sorted(self.entries, key=lambda x: x[0])  # 升序
        loss, epoch, path = sorted_h[0]
        return float(loss), int(epoch), str(path)


class ScoreTopKManager:
    """按 test_mean_score 降序保留 k 个 ckpt (PADP 源端行为,1:1 复刻)。"""

    def __init__(self,
                 save_dir: Path,
                 k: int = 5,
                 format_str: str = "epoch{epoch:03d}_score{test_mean_score:.4f}.ckpt"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.k = int(k)
        self.format_str = format_str
        # 维护 entries: [(score, epoch, path)],升序排列,头部是最差
        self.entries: list = []
        # 启动时扫描已有 ckpt,重建 entries(支持续训)
        self._reload_existing()

    def _reload_existing(self):
        """从磁盘 reload 已有 topk/*.ckpt, 重新建立 entries 列表(支持续训)。"""
        import re
        for ckpt_path in self.save_dir.glob("epoch*_score*.ckpt"):
            try:
                payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                score = float(payload.get("test_mean_score", 0.0))
                epoch = int(payload.get("epoch", 0))
                self.entries.append((score, epoch, str(ckpt_path)))
                logger.info("[ScoreTopK] 重新加载已有: %s score=%.4f", ckpt_path.name, score)
            except Exception as e:
                logger.warning("[ScoreTopK] 加载 %s 失败: %s", ckpt_path, e)
        # 排序:头部最差
        self.entries.sort(key=lambda x: x[0])

    def try_save(self, epoch: int, score: float, payload: dict) -> bool:
        """若 score 排进前 k 最大,则保存并返回 True。"""
        # payload 加 test_mean_score 字段(供 reload)
        payload["test_mean_score"] = float(score)
        payload["epoch"] = int(epoch)

        if len(self.entries) < self.k:
            should_save = True
        else:
            # 升序:头部最小(最差)
            worst_score = self.entries[0][0]
            should_save = score > worst_score

        if not should_save:
            return False

        ckpt_path = self.save_dir / self.format_str.format(epoch=epoch, test_mean_score=score)
        torch.save(payload, ckpt_path)
        logger.info("[ScoreTopK] 已保存 ckpt 到 %s (score=%.6f)", ckpt_path, score)
        self.entries.append((float(score), int(epoch), str(ckpt_path)))

        # 升序排序(头部最差)
        self.entries.sort(key=lambda x: x[0])
        # 若超出 k 个,删头部(最差)
        while len(self.entries) > self.k:
            _, _, old_path = self.entries.pop(0)
            try:
                Path(old_path).unlink()
                logger.info("[ScoreTopK] 删除分数更低的 ckpt: %s", old_path)
            except OSError as e:
                logger.warning("[ScoreTopK] 删除 %s 失败: %s", old_path, e)
        return True

    def best(self):
        """返回 (best_score, best_epoch, best_path)。空时返回 (None, None, None)。"""
        if not self.entries:
            return None, None, None
        sorted_h = sorted(self.entries, key=lambda x: x[0], reverse=True)
        score, epoch, path = sorted_h[0]
        return float(score), int(epoch), str(path)
