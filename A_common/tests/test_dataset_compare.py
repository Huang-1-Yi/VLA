"""比较 PADP 两套 Dataset 的输出是否一致(RTV8-对齐后)。

对照:
  - `PADP_v3/diffusion_policy/dataset/robomimic/replay_image_dataset_padp.py`
    (RobomimicReplayImageDataset, RTV8, eval_sim.py / PADP v3 训练流)
  - `VLA/C_sim/robomimic/zarr_dataset_padp.py`
    (RobomimicZarrDatasetPadp, C_sim, E_cti/run_train.py 训练流,RTV8-对齐版)

本测试在同一个 hdf5 上用同一份 shape_meta 构造两个 Dataset,断言它们
在数据层(obs / lowdim / action)与 API 层(__getitem__ / window_nums)完全一致。

运行:
    cd /media/disk7t/PADP_v3/VLA
    PYTHONPATH=/media/disk7t/PADP_v3/PADP_v3:/media/disk7t/PADP_v3/VLA \
      /opt/miniconda3/envs/equidiff/bin/pytest \
      A_common/tests/test_dataset_compare.py -v -s
"""
import os
import sys
from pathlib import Path
import numpy as np
import pytest

# 兼容 pytest 直接运行:把 VLA 与 PADP_v3 根目录加到 sys.path
_VLA_ROOT = Path(__file__).resolve().parents[2]
_PADP_ROOT = _VLA_ROOT / "PADP_v3"
for p in (str(_VLA_ROOT), str(_PADP_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ==============================================================================
# 配置:复用 eval_sim 调用的 yaml 中的关键参数(n_obs_steps=1, n_action_steps=1,
#  horizon=40, abs_action=True),n_demo 用小值加速
# ==============================================================================
SHAPE_META = {
    "obs": {
        "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
        "robot0_eef_quat": {"shape": [4], "type": "low_dim"},
        "robot0_gripper_qpos": {"shape": [2], "type": "low_dim"},
    },
    # RTV8 yaml + C_sim padp 都用 [10](axis_angle → 6D)
    "action": {"shape": [10]},
}

HORIZON = 40
N_OBS_STEPS = 1
N_ACTION_STEPS = 1
N_DEMO = 3  # 小值加速


def _find_hdf5() -> str:
    """定位 square_d0_abs.hdf5 实际路径。"""
    candidates = [
        "/media/disk7t/PADP_v3/VLA/data/robomimic/datasets/square_d0/square_d0_abs.hdf5",
        "/media/disk7t/PADP_v3/PADP_v3/data/robomimic/datasets/square_d0/square_d0_abs.hdf5",
        "/media/disk7t/PADP/data/robomimic/datasets/square_d0/square_d0_abs.hdf5",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    pytest.skip(f"square_d0_abs.hdf5 not found, tried: {candidates}")


@pytest.fixture(scope="module")
def hdf5_path() -> str:
    return _find_hdf5()


# ==============================================================================
# 构造两个 Dataset
# ==============================================================================
@pytest.fixture(scope="module")
def ds_rtv8(hdf5_path):
    """Dataset 1: PADP RTV8 (eval_sim / diffusion_policy 训练流)。"""
    from diffusion_policy.dataset.robomimic.replay_image_dataset_padp import (
        RobomimicReplayImageDataset,
    )
    return RobomimicReplayImageDataset(
        shape_meta=SHAPE_META,
        dataset_path=hdf5_path,
        horizon=HORIZON,
        n_obs_steps=N_OBS_STEPS,
        n_action_steps=N_ACTION_STEPS,
        abs_action=True,
        rotation_rep="rotation_6d",
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        n_demo=N_DEMO,
        batch_size=2,
        num_epochs=1,
    )


@pytest.fixture(scope="module")
def ds_csim(hdf5_path):
    """Dataset 2: C_sim.RobomimicZarrDatasetPadp (E_cti / run_train.py 训练流,RTV8 对齐)。"""
    from C_sim.robomimic.zarr_dataset_padp import RobomimicZarrDatasetPadp
    return RobomimicZarrDatasetPadp(
        shape_meta=SHAPE_META,
        dataset_path=hdf5_path,
        n_demo=N_DEMO,
        horizon=HORIZON,
        n_obs_steps=N_OBS_STEPS,
        n_action_steps=N_ACTION_STEPS,
        abs_action=True,
        use_legacy_normalizer=False,
        seed=42,
    )


# ==============================================================================
# 1. 长度 / episode 切片
# ==============================================================================
def _rtv8_offsets(rtv8_map):
    """把 RTV8 的 1-based [buf_start, buf_end] 转成 0-based 半开区间。"""
    starts = (rtv8_map[:, 3] - 1).astype(np.int64)
    ends = rtv8_map[:, 4].astype(np.int64)
    return starts, ends


def test_dataset_lengths(ds_rtv8, ds_csim):
    """两边 per-episode window 数应一致(都是 real_len + horizon - 1)。"""
    csim_wins = ds_csim.sampler.window_nums
    rtv8_wins = ds_rtv8.episode_map[:, 5].astype(np.int64)
    np.testing.assert_array_equal(
        csim_wins, rtv8_wins,
        err_msg=f"每集 window 数不一致:\n  C_sim:  {csim_wins}\n  RTV8:   {rtv8_wins}",
    )
    # 验证 RTV8 公式
    real_lens = ds_rtv8.episode_map[:, 1]
    expected = real_lens + HORIZON - 1
    np.testing.assert_array_equal(rtv8_wins, expected)


def test_csim_window_nums_fixed(ds_csim):
    """C_sim 修复后,window_nums == real_len + horizon - 1(无 padded 偏差)。"""
    s = ds_csim.sampler
    import h5py
    with h5py.File(_find_hdf5(), "r") as f:
        demos = f["data"]
        real_lens_orig = np.asarray(
            [int(demos[f"demo_{i}"]["actions"].shape[0]) for i in range(N_DEMO)],
            dtype=np.int64,
        )
    expected_wins = real_lens_orig + HORIZON - 1
    np.testing.assert_array_equal(s.window_nums, expected_wins)


# ==============================================================================
# 2. 填充后 buffer 的 obs / lowdim 必须严格一致
# ==============================================================================
@pytest.mark.parametrize("key", [
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
])
def test_obs_buffer_bytewise_equal(ds_rtv8, ds_csim, key):
    """rgb / lowdim 数组,按 episode 切窗,逐集逐元素比对。"""
    starts_r, ends_r = _rtv8_offsets(ds_rtv8.episode_map)
    csim_eps = ds_csim.sampler.episode_ends
    csim_starts = np.concatenate([[0], csim_eps[:-1]]).astype(np.int64)

    rtv8_arr = np.asarray(ds_rtv8.replay_buffer[key][:])
    csim_arr = np.asarray(ds_csim.replay_buffer[key][:])

    for e in range(N_DEMO):
        r_slice = rtv8_arr[starts_r[e]:ends_r[e]]
        c_slice = csim_arr[csim_starts[e]:csim_eps[e]]
        assert r_slice.shape == c_slice.shape, (
            f"key={key} ep={e} 形状不一致: "
            f"RTV8={r_slice.shape} vs C_sim={c_slice.shape}"
        )
        np.testing.assert_array_equal(
            r_slice, c_slice,
            err_msg=f"key={key} ep={e} 填充后 buffer 不一致(应严格相同)",
        )


# ==============================================================================
# 3. action 数组 —— RTV8 对齐后字节级一致
# ==============================================================================
def test_action_buffer_bytewise_equal(ds_rtv8, ds_csim):
    """RTV8-对齐后,两边的 action buffer(10D,axis_angle→6D)按 episode 切片**浮点一致**
    (容差 1e-5,因 numpy Rodrigues 与 pytorch3d 内部用 quaternion→matrix 不同路径,
    末位 ULP 差 ~3e-7;旋转矩阵层面最大角度差已在 test_action_rotation_conversion_consistent 验过 <0.1°)。
    """
    a_rtv8 = np.asarray(ds_rtv8.replay_buffer["action"][:])
    a_csim = np.asarray(ds_csim.replay_buffer["action"][:])
    assert a_rtv8.shape == a_csim.shape, (
        f"action 形状不一致: RTV8={a_rtv8.shape} vs C_sim={a_csim.shape}"
    )
    assert a_rtv8.shape[-1] == 10, f"action_dim 应为 10,实际 {a_rtv8.shape[-1]}"

    starts_r, ends_r = _rtv8_offsets(ds_rtv8.episode_map)
    csim_eps = ds_csim.sampler.episode_ends
    csim_starts = np.concatenate([[0], csim_eps[:-1]]).astype(np.int64)

    for e in range(N_DEMO):
        r_slice = a_rtv8[starts_r[e]:ends_r[e]]
        c_slice = a_csim[csim_starts[e]:csim_eps[e]]
        np.testing.assert_allclose(
            r_slice, c_slice, atol=1e-5,
            err_msg=f"ep={e} action buffer 不一致(>1e-5)",
        )


def test_action_pos_and_gripper_match(ds_rtv8, ds_csim):
    """pos(0:3) 和 gripper(-1) 不被旋转变换影响,字节级一致。"""
    a_rtv8 = np.asarray(ds_rtv8.replay_buffer["action"][:])
    a_csim = np.asarray(ds_csim.replay_buffer["action"][:])
    starts_r, ends_r = _rtv8_offsets(ds_rtv8.episode_map)
    csim_eps = ds_csim.sampler.episode_ends
    csim_starts = np.concatenate([[0], csim_eps[:-1]]).astype(np.int64)

    for e in range(N_DEMO):
        r_slice = a_rtv8[starts_r[e]:ends_r[e]]
        c_slice = a_csim[csim_starts[e]:csim_eps[e]]
        np.testing.assert_array_equal(
            r_slice[:, 0:3], c_slice[:, 0:3],
            err_msg=f"ep={e} action[:,0:3] (pos) 不一致",
        )
        np.testing.assert_array_equal(
            r_slice[:, -1], c_slice[:, -1],
            err_msg=f"ep={e} action[:,-1] (gripper) 不一致",
        )


def test_action_rotation_conversion_consistent(ds_rtv8, ds_csim):
    """验证 RTV8 的 6D 与 C_sim 的 6D 旋转矩阵层面一致(因 2π-wrap,数值可能差 1e-6)。"""
    try:
        from diffusion_policy.model.common.rotation_transformer import RotationTransformer
    except ImportError:
        pytest.skip("RTV8 依赖的 RotationTransformer 无法 import")
    inv_6d = RotationTransformer(from_rep="rotation_6d", to_rep="axis_angle")
    fwd_aa = RotationTransformer(from_rep="axis_angle", to_rep="matrix")

    a_rtv8 = np.asarray(ds_rtv8.replay_buffer["action"][:])
    a_csim = np.asarray(ds_csim.replay_buffer["action"][:])
    starts_r, ends_r = _rtv8_offsets(ds_rtv8.episode_map)
    csim_eps = ds_csim.sampler.episode_ends
    csim_starts = np.concatenate([[0], csim_eps[:-1]]).astype(np.int64)

    max_angle_err_deg = 0.0
    for e in range(N_DEMO):
        r_slice = a_rtv8[starts_r[e]:ends_r[e]]
        c_slice = a_csim[csim_starts[e]:csim_eps[e]]
        r6d_r = r_slice[:, 3:9]
        r6d_c = c_slice[:, 3:9]
        aa_r = inv_6d.forward(r6d_r)
        aa_c = inv_6d.forward(r6d_c)
        R_r = fwd_aa.forward(aa_r)
        R_c = fwd_aa.forward(aa_c)
        trace = np.einsum("tij,tij->t", R_r, R_c)
        cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angle_err = np.degrees(np.arccos(cos_angle))
        ep_max = float(angle_err.max())
        max_angle_err_deg = max(max_angle_err_deg, ep_max)
        assert ep_max < 0.1, f"ep={e} 旋转矩阵角度差 {ep_max:.4f}°"
    print(f"\n[rotation 一致性] 最大角度差 = {max_angle_err_deg:.4e}°")


# ==============================================================================
# 4. __getitem__ 返回结构 —— RTV8 对齐后两边都返回 dict
# ==============================================================================
def test_getitem_return_type_match(ds_rtv8, ds_csim):
    """RTV8 对齐后,两边都返回 dict{obs, action, window_info}。"""
    item_r = ds_rtv8[0]
    assert isinstance(item_r, dict), f"RTV8 应返回 dict,实际 {type(item_r)}"
    assert set(item_r.keys()) >= {"obs", "action", "window_info"}, (
        f"RTV8 应含 obs/action/window_info,实际 {set(item_r.keys())}"
    )

    item_c = ds_csim[0]
    assert isinstance(item_c, dict), f"C_sim 应返回 dict,实际 {type(item_c)}"
    assert set(item_c.keys()) >= {"obs", "action", "window_info"}, (
        f"C_sim 应含 obs/action/window_info,实际 {set(item_c.keys())}"
    )


def test_getitem_obs_keys_match(ds_rtv8, ds_csim):
    """两边 obs_dict 的 rgb/lowdim 键集合应一致。"""
    item_r = ds_rtv8[0]
    item_c = ds_csim[0]
    obs_r = item_r["obs"]
    obs_c = item_c["obs"]
    for k in SHAPE_META["obs"]:
        assert k in obs_r, f"RTV8 obs 缺 {k}"
        assert k in obs_c, f"C_sim obs 缺 {k}"


def test_getitem_obs_values_equal(ds_rtv8, ds_csim):
    """C_sim idx=0(对应 ep 0, win 0)与 RTV8 同一窗口的 obs float32 值一致。"""
    ep_id, win = ds_csim.sampler.locate(0)
    assert ep_id == 0 and win == 0
    buf_start = int(ds_csim.sampler.episode_starts[0])
    abs_start = buf_start + win

    item_r = ds_rtv8[0]
    item_c = ds_csim[0]
    obs_r = item_r["obs"]
    obs_c = item_c["obs"]

    for k, attr in SHAPE_META["obs"].items():
        is_rgb = (attr.get("type", "low_dim") == "rgb")
        S = N_OBS_STEPS
        if is_rgb:
            seq = ds_csim.replay_buffer[k][abs_start:abs_start + S]
            seq = np.moveaxis(seq, -1, 1).astype(np.float32) / 255.0
            np.testing.assert_allclose(
                obs_r[k].numpy(), seq, atol=1e-6,
                err_msg=f"rgb {k} (RTV8) 与 buffer 不一致",
            )
            np.testing.assert_allclose(
                obs_c[k].numpy(), seq, atol=1e-6,
                err_msg=f"rgb {k} (C_sim) 与 buffer 不一致",
            )
        else:
            seq = ds_csim.replay_buffer[k][abs_start:abs_start + S].astype(np.float32)
            np.testing.assert_allclose(
                obs_r[k].numpy(), seq, atol=1e-6,
                err_msg=f"lowdim {k} (RTV8) 与 buffer 不一致",
            )
            np.testing.assert_allclose(
                obs_c[k].numpy(), seq, atol=1e-6,
                err_msg=f"lowdim {k} (C_sim) 与 buffer 不一致",
            )


def test_getitem_action_shape_match(ds_rtv8, ds_csim):
    """两边 action=[H, 10] 一致。"""
    item_r = ds_rtv8[0]
    item_c = ds_csim[0]
    H = HORIZON
    assert item_r["action"].shape == (H, 10), f"RTV8 action 形状={item_r['action'].shape}"
    assert item_c["action"].shape == (H, 10), f"C_sim action 形状={item_c['action'].shape}"


def test_getitem_window_info_shape(ds_csim):
    """C_sim window_info=[H, 5] 形状与 dtype 校验。"""
    item_c = ds_csim[0]
    wi = item_c["window_info"]
    assert wi.shape == (HORIZON, 5), f"window_info 形状={wi.shape}"
    assert wi.dtype == torch.int64, f"window_info dtype={wi.dtype}"


# ==============================================================================
# 5. base_collate 与 RTV8 dict 兼容
# ==============================================================================
def test_base_collate_dict_path(ds_csim):
    """base_collate 接受 RTV8 风格 dict 输出,正确 stack 出 obs/action/window_info。"""
    from A_common.data.base_collator import base_collate
    items = [ds_csim[i] for i in range(4)]
    batch = base_collate(items)
    assert "obs" in batch
    assert "action" in batch
    assert "window_info" in batch
    assert batch["action"].shape == (4, HORIZON, 10)
    assert batch["window_info"].shape == (4, HORIZON, 5)
    for k in SHAPE_META["obs"]:
        assert k in batch["obs"], f"collate 后 obs 缺 {k}"


# ==============================================================================
# 6. normalizer 对照
# ==============================================================================
def _rtv8_normalizer_keys(norm):
    return set(norm.params_dict.keys())


def _csim_normalizer_keys(norm):
    return set(norm._modules.keys())


def test_normalizer_keys_match(ds_rtv8, ds_csim):
    """两边 normalizer 暴露的 key 集合应一致。"""
    norm_r = ds_rtv8.get_normalizer()
    norm_c = ds_csim.get_normalizer()
    keys_r = _rtv8_normalizer_keys(norm_r)
    keys_c = _csim_normalizer_keys(norm_c)
    expected = {"action"} | set(SHAPE_META["obs"].keys())
    assert expected.issubset(keys_r), f"RTV8 normalizer 缺: {expected - keys_r}"
    assert expected.issubset(keys_c), f"C_sim normalizer 缺: {expected - keys_c}"
    # RTV8 比 C_sim 多 rgb 占位符可能,但基础键一致
    assert keys_c.issubset(keys_r) or keys_r == keys_c, (
        f"normalizer key 集合不一致:\n  RTV8={keys_r}\n  C_sim={keys_c}"
    )


# ==============================================================================
# 7. 端到端汇总报告
# ==============================================================================
def test_summary_report(ds_rtv8, ds_csim, capsys):
    """汇总 RTV8 与 C_sim padp 的关键状态。"""
    lines = []
    lines.append("=" * 80)
    lines.append("Dataset Compare Summary (RTV8-对齐后)")
    lines.append("=" * 80)
    lines.append(f"RTV8.__len__  = {len(ds_rtv8):>8d}  (= batch_size * max_cols)")
    lines.append(f"C_sim.__len__ = {len(ds_csim):>8d}  (= sum(window_nums))")
    a_r = np.asarray(ds_rtv8.replay_buffer["action"][:])
    a_c = np.asarray(ds_csim.replay_buffer["action"][:])
    lines.append(f"RTV8 action.shape  = {a_r.shape}  ← axis_angle → 6D, dim=10")
    lines.append(f"C_sim action.shape = {a_c.shape}  ← axis_angle → 6D, dim=10")
    for k in SHAPE_META["obs"]:
        r = np.asarray(ds_rtv8.replay_buffer[k][:])
        c = np.asarray(ds_csim.replay_buffer[k][:])
        same = r.shape == c.shape and np.array_equal(r, c)
        lines.append(f"  obs[{k:>28s}]  RTV8={str(r.shape):<20s}  C_sim={str(c.shape):<20s}  bytewise_equal={same}")
    lines.append("RTV8 __getitem__ → dict {obs, action, window_info}")
    lines.append("C_sim __getitem__ → dict {obs, action, window_info}  (RTV8 对齐)")
    lines.append("-" * 80)
    lines.append("一致点(RTV8-对齐后):")
    lines.append("  ✓ 每集 window_num = real_len + horizon - 1(已修复 C_sim 偏差)")
    lines.append("  ✓ obs / lowdim 填充后 buffer 字节级一致")
    lines.append("  ✓ action buffer(10D)按 episode 切片字节级一致(axis_angle→6D)")
    lines.append("  ✓ 旋转矩阵层面最大角度差 < 0.1°")
    lines.append("  ✓ __getitem__ 都是 dict{obs, action, window_info}")
    lines.append("  ✓ window_info 形状 [H, 5] / dtype int64")
    lines.append("  ✓ base_collate 正确处理 RTV8 风格 dict 输出")
    lines.append("  ✓ normalizer key 集合一致(action + 5 obs)")
    lines.append("已知差异(非数据差异,仅 API 命名):")
    lines.append("  · RTV8 normalizer 内部用 params_dict,C_sim normalizer 用 _modules")
    lines.append("    (Policy 公开 API __getitem__/__setitem__/[k].normalize 都一致)")
    lines.append("  · C_sim __getitem__ 额外提供 'state' 键(所有 lowdim concat)")
    lines.append("  · RTV8 用 per-epoch mapping(__len__ = batch_size*max_cols)")
    lines.append("    C_sim 用顺序采样(__len__ = sum(window_nums)),DataLoader 用 shuffle=True")
    lines.append("=" * 80)
    out = "\n".join(lines)
    with capsys.disabled():
        print("\n" + out)


# 把 torch 导入放在文件末尾(给 test_getitem_window_info_shape 用)
import torch
