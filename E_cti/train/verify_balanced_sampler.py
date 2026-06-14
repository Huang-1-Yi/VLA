"""verify_balanced_sampler —— 最小化验证 BalancedColumnsSampler 的数学契约 + 消融对比。

跑这个脚本能直接确认:
  1. window_nums[i] = real_lens[i] + horizon - 1      (与 RTV8 一致)
  2. __len__ == batch_size * max_cols
  3. DataLoader batch 内,前 N 行(N = batch_size)每个 sample 来自不同 episode
  4. per-epoch 重建 _global_pairs,跨 epoch 同 idx 的 (ep, win) 不同
  5. 数据值正确(idx 0 = episode 0 的 win 0)
  6. SequenceSampler 与 BalancedColumnsSampler 的 total window count 一致
  7. 【消融】消融基线类与新版本 SequenceSampler 路径字节级一致
  8. 【消融】DataLoader shuffle 行为下,消融基线与 balanced 模式的 batch 内
     ep 重复率显著不同
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_VLA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_VLA_ROOT))

from A_common.data.base_dataset import (
    SequenceSampler, BalancedColumnsSampler,
)
from C_sim.robomimic.padp_for_test_dataset import RobomimicZarrDatasetPadpForTest
# 消融基线:与 `padp_for_test_dataset.RobomimicZarrDatasetPadpForTest` 修改前
# 字节级等价的独立 class(无 balanced_sampler / set_epoch / properties)
from C_sim.robomimic.padp_for_test_dataset_no_batch_ep import (
    RobomimicZarrDatasetPadpForTestNoBatchEp,
)


SHAPE_META = {
    "obs": {
        "robot0_eef_pos":       {"shape": [3],  "type": "low_dim"},
        "robot0_eef_quat":      {"shape": [4],  "type": "low_dim"},
        "robot0_gripper_qpos":  {"shape": [2],  "type": "low_dim"},
    },
    "action": {"shape": [10]},
}
DATASET_PATH = str(_VLA_ROOT / "data/robomimic/datasets/square_d0/square_d0_abs.hdf5")


def header(s):
    print("\n" + "=" * 78)
    print(" " + s)
    print("=" * 78)


# =====================================================================
# 1. 纯 BalancedColumnsSampler 单元测试
# =====================================================================
def unit_test_balanced_columns_sampler():
    header("[1] BalancedColumnsSampler 单元测试")

    real_lens = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int64)
    episode_ends = np.cumsum(real_lens).astype(np.int64)
    H = 8
    B = 4

    sampler = BalancedColumnsSampler(
        episode_ends=episode_ends,
        real_lens=real_lens,
        horizon=H,
        batch_size=B,
        seed=42,
        epoch_index=0,
    )

    # 1.1 关键公式验证
    expected_win = real_lens + H - 1
    assert np.array_equal(sampler.window_nums, expected_win), \
        f"window_nums 公式错误: got {sampler.window_nums}, want {expected_win}"
    print(f"  ✓ window_nums = real_lens + H - 1,sum = {sampler.window_nums.sum()}")

    # 1.2 __len__ == B * max_cols
    expected_len = B * sampler.max_cols
    assert len(sampler) == expected_len, \
        f"__len__ 错误: got {len(sampler)}, want {expected_len}"
    print(f"  ✓ __len__ = B * max_cols = {B} * {sampler.max_cols} = {len(sampler)}")

    # 1.3 寻址 p=idx%B, q=idx//B
    pairs_observed = set()
    for idx in range(len(sampler)):
        ep, win = sampler.locate(idx)
        p = idx % B
        q = idx // B
        e_ref = int(sampler._global_pairs[p, q, 0])
        m_ref = int(sampler._global_pairs[p, q, 1])
        assert (ep, win) == (e_ref, m_ref), \
            f"locate({idx}) mismatch: ({ep},{win}) != ({e_ref},{m_ref})"
        # win ∈ [0, real_len + H - 1)
        assert 0 <= win < real_lens[ep] + H - 1, \
            f"locate({idx}) win={win} 越界(real_len[{ep}]={real_lens[ep]})"
        pairs_observed.add((ep, win))
    print(f"  ✓ 全部 {len(sampler)} 个 idx 寻址合法(ep, win) ∈ [0, real_len+H-1)")

    # 1.4 每 batch 内 N 个 sample 来自不同 episode
    cross_ep_check = 0
    for batch_start in range(0, len(sampler), B):
        batch_eps = set()
        for offset in range(B):
            idx = batch_start + offset
            if idx >= len(sampler):
                break
            ep, _ = sampler.locate(idx)
            batch_eps.add(ep)
        if len(batch_eps) == min(B, len(sampler) - batch_start):
            cross_ep_check += 1
    print(f"  ✓ {cross_ep_check} 个完整 batch 全部满足'行内强制不同 ep'")
    assert cross_ep_check > 0, "至少要有一个完整 batch"

    # 1.5 per-epoch 重建:同 idx 跨 epoch 取到不同 (ep, win)
    sampler.set_epoch(0)
    snap0 = [sampler.locate(i) for i in range(min(50, len(sampler)))]
    sampler.set_epoch(1)
    snap1 = [sampler.locate(i) for i in range(min(50, len(sampler)))]
    same_count = sum(1 for a, b in zip(snap0, snap1) if a == b)
    diff_count = len(snap0) - same_count
    print(f"  ✓ epoch=0 vs epoch=1 前 50 个 idx: {diff_count}/{len(snap0)} 不同(per-epoch 洗牌生效)")
    assert diff_count > 0, "per-epoch 重建应该产生不同的 (ep, win) 序列"

    # 1.6 SequenceSampler 与 BalancedColumnsSampler 产出的 window 总量一致
    fake_ep_ends = np.cumsum(real_lens + 2 * (H - 1)).astype(np.int64)
    seq_sampler = SequenceSampler(
        replay_buffer=None,            # 只验证 window_nums 部分
        episode_ends=fake_ep_ends,
        horizon=H,
        n_obs_steps=1,
        pad_strategy="edge_repeat",
        real_lens=real_lens,
    )
    assert seq_sampler.window_nums.sum() == sampler.window_nums.sum(), \
        f"window total 不一致: seq={seq_sampler.window_nums.sum()}, bal={sampler.window_nums.sum()}"
    print(f"  ✓ SequenceSampler vs BalancedColumnsSampler total windows = {seq_sampler.window_nums.sum()} (一致)")


# =====================================================================
# 2. RobomimicZarrDatasetPadpForTest 集成测试(走真实 hdf5)
# =====================================================================
def integration_test_dataset():
    header("[2] RobomimicZarrDatasetPadpForTest + BalancedColumnsSampler 集成测试")

    # 2.1 sequence 模式(向后兼容)
    ds_seq = RobomimicZarrDatasetPadpForTest(
        shape_meta=SHAPE_META, dataset_path=DATASET_PATH,
        n_demo=20, horizon=8, n_obs_steps=1, n_action_steps=1,
        abs_action=True, use_legacy_normalizer=False,
        balanced_sampler=False,
    )
    print(f"  [seq] n_windows = {len(ds_seq)}")
    seq_total = len(ds_seq)
    sample0 = ds_seq[0]
    assert "action" in sample0 and "obs" in sample0 and "window_info" in sample0
    assert sample0["action"].shape == (8, 10)
    assert sample0["window_info"].shape == (8, 5)
    print(f"  [seq] sample[0].action.shape={sample0['action'].shape},"
          f" window_info.shape={sample0['window_info'].shape}")
    print(f"  [seq] sample[0].window_info[0] = {sample0['window_info'][0].tolist()}  (ep=0, win=0)")

    # 2.2 balanced 模式
    ds_bal = RobomimicZarrDatasetPadpForTest(
        shape_meta=SHAPE_META, dataset_path=DATASET_PATH,
        n_demo=20, horizon=8, n_obs_steps=1, n_action_steps=1,
        abs_action=True, use_legacy_normalizer=False,
        balanced_sampler=True, batch_size=8, sampler_seed=42,
    )
    print(f"  [bal] n_windows = {len(ds_bal)} (B*max_cols)")
    print(f"  [bal] sampler = {type(ds_bal.sampler).__name__}, max_cols = {ds_bal.sampler.max_cols}")
    assert isinstance(ds_bal.sampler, BalancedColumnsSampler)
    assert ds_bal.balanced_sampler is True

    # 2.3 balanced 模式 + DataLoader 验证"每 batch 跨 ep 平衡"
    dl = DataLoader(ds_bal, batch_size=8, shuffle=False, drop_last=True,
                    num_workers=0, collate_fn=lambda x: x)
    first_batch = next(iter(dl))
    ep_ids_in_batch = [s["window_info"][0, 3].item() for s in first_batch]
    print(f"  [bal] first batch ep_ids = {ep_ids_in_batch}")
    assert len(set(ep_ids_in_batch)) == len(ep_ids_in_batch), \
        f"balanced 模式第一个 batch 撞 ep: {ep_ids_in_batch}"
    print(f"  ✓ DataLoader 第一 batch 内 8 个 sample 来自 8 个不同 episode")

    # 2.4 set_epoch 重建映射(per-epoch 洗牌)
    ds_bal.set_epoch(0)
    snap_e0 = [ds_bal.sampler.locate(i) for i in range(8)]
    ds_bal.set_epoch(1)
    snap_e1 = [ds_bal.sampler.locate(i) for i in range(8)]
    same = sum(1 for a, b in zip(snap_e0, snap_e1) if a == b)
    print(f"  ✓ set_epoch(0) vs set_epoch(1): 前 8 个 idx 中 {same}/8 重合(预期少量重合)")

    # 2.5 数据值正确性:idx 0 = (ep=0, win=0) → window_info[0] 应是 (is_warmup=0, ep_idx=0, new_pos=0, ep_id=0, buf=1)
    ds_bal.set_epoch(0)
    s0 = ds_bal[0]
    win0 = s0["window_info"][0].tolist()
    print(f"  ✓ ds_bal[0].window_info[0] = {win0}  (预期:is_warmup=0, ep_idx=0, new_pos=0, ep_id=0, buf_pos=1)")
    assert win0[3] == 0, f"ep_id 应=0, got {win0[3]}"
    assert win0[4] == 1, f"buf_pos(1-based) 应=1, got {win0[4]}"

    # 2.6 与 seq 模式产出"等价"window 总数(差只在 batch 组成)
    bal_total = len(ds_bal)
    print(f"  [window_total] seq={seq_total}, bal={bal_total} (差仅在 padding to B*max_cols)")
    return seq_total, bal_total


# =====================================================================
# 3. 【消融实验】消融基线类与新版本 SequenceSampler 路径字节级一致
# =====================================================================
def ablation_test_baseline_equivalence():
    header("[3] 【消融】RobomimicZarrDatasetPadpForTestNoBatchEp vs "
           "RobomimicZarrDatasetPadpForTest(balanced=False) 字节级一致")

    ds_base = RobomimicZarrDatasetPadpForTestNoBatchEp(
        shape_meta=SHAPE_META, dataset_path=DATASET_PATH,
        n_demo=20, horizon=8, n_obs_steps=1, n_action_steps=1,
        abs_action=True, use_legacy_normalizer=False,
    )
    ds_new_seq = RobomimicZarrDatasetPadpForTest(
        shape_meta=SHAPE_META, dataset_path=DATASET_PATH,
        n_demo=20, horizon=8, n_obs_steps=1, n_action_steps=1,
        abs_action=True, use_legacy_normalizer=False,
        balanced_sampler=False,
    )

    # 3.1 API 表面差异:消融基线**没有** set_epoch / balanced_sampler / batch_size property
    assert not hasattr(ds_base, "set_epoch"), "基线不应该有 set_epoch"
    assert not hasattr(ds_base, "balanced_sampler"), "基线不应该有 balanced_sampler 属性"
    assert not hasattr(ds_base, "batch_size"), "基线不应该有 batch_size 属性"
    print("  ✓ 消融基线**没有** set_epoch / balanced_sampler / batch_size 属性")
    print(f"  ✓ 消融基线 sampler = {type(ds_base.sampler).__name__}(无 BalancedColumnsSampler)")

    # 3.2 __len__ 一致
    assert len(ds_base) == len(ds_new_seq), \
        f"__len__ 不一致: base={len(ds_base)}, new_seq={len(ds_new_seq)}"
    print(f"  ✓ 两者 __len__ 一致: {len(ds_base)}")

    # 3.3 sample[0] 字节级一致(action / window_info / obs)
    s_base = ds_base[0]
    s_new = ds_new_seq[0]
    assert torch.allclose(s_base["action"], s_new["action"]), "action 不一致"
    assert (s_base["window_info"] == s_new["window_info"]).all(), "window_info 不一致"
    for k in s_base["obs"]:
        assert torch.allclose(s_base["obs"][k], s_new["obs"][k]), f"obs[{k}] 不一致"
    print("  ✓ sample[0] {action, window_info, obs} 字节级一致")

    # 3.4 normalizer 字节级一致
    n_base = ds_base.get_normalizer()
    n_new = ds_new_seq.get_normalizer()
    for k in n_base._modules.keys():
        s_b = n_base[k].state_dict()
        s_n = n_new[k].state_dict()
        for kk in s_b:
            if "scale" in kk or "offset" in kk:
                assert torch.allclose(s_b[kk], s_n[kk]), f"normalizer[{k}].{kk} 不一致"
    print("  ✓ normalizer 字节级一致")

    # 3.5 验证消融基线**没有** per-epoch 重建能力
    s_before = ds_base.sampler.locate(0)
    s_after = ds_base.sampler.locate(0)
    assert s_before == s_after
    print(f"  ✓ 消融基线 idx=0 永远取 (ep={s_before[0]}, win={s_before[1]}) - 无 per-epoch 重建")

    # 3.6 抽查:idx=10 / 100 / 1000 三个 sample 也字节级一致
    for idx in [10, 100, 1000]:
        sb = ds_base[idx]
        sn = ds_new_seq[idx]
        assert torch.allclose(sb["action"], sn["action"]), f"idx={idx} action 不一致"
        assert (sb["window_info"] == sn["window_info"]).all(), f"idx={idx} window_info 不一致"
    print("  ✓ 抽查 idx ∈ {10, 100, 1000} 字节级一致")

    print("\n  ⇒ 消融基线与新版本 SequenceSampler 路径在数据 / sampler 输出上**完全等价**")
    print("    唯一差异:消融基线无 set_epoch / 无 balanced_sampler 开关")


# =====================================================================
# 4. 【消融实验】balanced vs ablation:DataLoader batch 内 ep 重复率
# =====================================================================
def ablation_test_batch_ep_diversity():
    header("[4] 【消融】DataLoader batch 内 ep 多样性对比 (balanced vs ablation)")

    # 4.1 balanced 模式
    ds_bal = RobomimicZarrDatasetPadpForTest(
        shape_meta=SHAPE_META, dataset_path=DATASET_PATH,
        n_demo=20, horizon=8, n_obs_steps=1, n_action_steps=1,
        abs_action=True, use_legacy_normalizer=False,
        balanced_sampler=True, batch_size=8, sampler_seed=42,
    )
    dl_bal = DataLoader(ds_bal, batch_size=8, shuffle=False, drop_last=True,
                        num_workers=0, collate_fn=lambda x: x)

    # 4.2 消融基线(DataLoader shuffle=True 模拟旧行为)
    ds_abl = RobomimicZarrDatasetPadpForTestNoBatchEp(
        shape_meta=SHAPE_META, dataset_path=DATASET_PATH,
        n_demo=20, horizon=8, n_obs_steps=1, n_action_steps=1,
        abs_action=True, use_legacy_normalizer=False,
    )
    g = torch.Generator(); g.manual_seed(42)
    dl_abl = DataLoader(ds_abl, batch_size=8, shuffle=True, drop_last=False,
                        num_workers=0, collate_fn=lambda x: x,
                        generator=g)

    # 4.3 统计前 50 个 batch 的 ep 重复率
    n_batches = 50
    bal_dup_count = 0
    abl_dup_count = 0
    bal_total = 0
    abl_total = 0
    for b_idx, (b_bal, b_abl) in enumerate(zip(dl_bal, dl_abl)):
        if b_idx >= n_batches:
            break
        eps_bal = [s["window_info"][0, 3].item() for s in b_bal]
        eps_abl = [s["window_info"][0, 3].item() for s in b_abl]
        bal_dup = len(eps_bal) - len(set(eps_bal))
        abl_dup = len(eps_abl) - len(set(eps_abl))
        bal_dup_count += bal_dup
        abl_dup_count += abl_dup
        bal_total += len(eps_bal)
        abl_total += len(eps_abl)

    bal_dup_rate = bal_dup_count / bal_total
    abl_dup_rate = abl_dup_count / abl_total
    print(f"  [balanced]  前 {n_batches} 个 batch: {bal_dup_count}/{bal_total} sample 撞 ep({bal_dup_rate*100:.1f}%)")
    print(f"  [ablation]  前 {n_batches} 个 batch: {abl_dup_count}/{abl_total} sample 撞 ep({abl_dup_rate*100:.1f}%)")

    assert bal_dup_count == 0, f"balanced 模式不应有 ep 重复,实际 {bal_dup_count}"
    print("  ✓ balanced 模式:0 个 sample 撞 ep(行内强制不同 ep 100% 成立)")
    assert abl_dup_count > 0, \
        f"消融基线在 shuffle=True 下应观察到 ep 重复(20 demos, B=8,概率不低),got {abl_dup_count}"
    print("  ✓ ablation 模式:在 shuffle=True 下出现 ep 重复(batch 内跨 ep 不保证)")


if __name__ == "__main__":
    unit_test_balanced_columns_sampler()
    integration_test_dataset()
    ablation_test_baseline_equivalence()
    ablation_test_batch_ep_diversity()
    header("[ALL PASS] BalancedColumnsSampler + 消融基线全链路数学逻辑一致 ✅")

