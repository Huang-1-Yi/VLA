# ============================================================
# PADP-VLA v1.0
# 1.0 版本,可训练 PADP 但无 rollout
# ============================================================

"""基础 collate 函数:把 list[(obs_dict, action)] 拼成 batch。"""
import torch


def base_collate(batch):
    """list of dataset samples → dict of batched tensors。

    支持两种 dataset 输出契约(自动嗅探):
      - RTV8 风格 dict:{obs, action, window_info?}
      - C_sim legacy tuple:(obs_dict, action)

    obs_dict 处理:
        - rgb 子字典每个 key 各自 stack → [B, S, C, H, W]
        - state 整体 stack → [B, S, D_s]
    action:      [B, H, D_a]
    window_info: [B, H, 5]   (RTV8 风格,无则省略)
    """
    # ===== RTV8 风格:batch[0] 是 dict =====
    if len(batch) > 0 and isinstance(batch[0], dict):
        out_obs = {}
        keys = set()
        for item in batch:
            keys.update(item["obs"].keys())
        for k in keys:
            vals = [it["obs"][k] for it in batch if k in it["obs"]]
            if not vals:
                continue
            out_obs[k] = torch.stack(vals, dim=0) if isinstance(vals[0], torch.Tensor) else vals

        out = {"obs": out_obs, "action": torch.stack([it["action"] for it in batch], dim=0)}
        # 透传 window_info(RTV8 必有,RobomimicZarrDatasetPadp 必有)
        if all("window_info" in it for it in batch):
            out["window_info"] = torch.stack([it["window_info"] for it in batch], dim=0)
        return out

    # ===== C_sim legacy tuple path(向后兼容)=====
    obs_dicts = [item[0] for item in batch]
    actions = torch.stack([item[1] for item in batch], dim=0)  # [B, H, D_a]

    out_obs = {}
    # 收集所有 key
    keys = set()
    for od in obs_dicts:
        keys.update(od.keys())

    for k in keys:
        # 跳过缺失的样本
        vals = [od[k] for od in obs_dicts if k in od]
        if not vals:
            continue
        if isinstance(vals[0], torch.Tensor):
            out_obs[k] = torch.stack(vals, dim=0)
        else:
            out_obs[k] = vals  # 留给上层处理

    return {"obs": out_obs, "action": actions}
