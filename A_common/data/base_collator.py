"""基础 collate 函数:把 list[(obs_dict, action)] 拼成 batch。"""
import torch


def base_collate(batch):
    """list of (obs_dict, action) → dict of batched tensors。

    obs_dict 处理:
        - rgb 子字典每个 key 各自 stack → [B, S, C, H, W]
        - state 整体 stack → [B, S, D_s]
    action:  [B, H, D_a]
    """
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
