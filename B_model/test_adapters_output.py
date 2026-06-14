# -*- coding: utf-8 -*-
"""B_model 自动 smoke test:遍历 adapters/ 下所有具体 Adapter 类(非抽象 BaseAdapter),
实例化并跑一次 forward,打印输出 shape 和参数量。

运行:
    cd VLA && python B_model/test_adapters_output.py
"""
import os
import sys
import inspect
import importlib
from pathlib import Path

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
VLA_ROOT = THIS_DIR.parent
sys.path.insert(0, str(VLA_ROOT))

from B_model.adapters.base_adapter import BaseAdapter  # noqa: E402


def _is_concrete_module(cls):
    if not inspect.isclass(cls):
        return False
    if not issubclass(cls, nn.Module):
        return False
    if inspect.isabstract(cls):
        return False
    # 跳过抽象基类(BaseXxx / AbstractXxx / xxxBase)
    if cls.__name__.startswith("Base") or cls.__name__.startswith("Abstract"):
        return False
    return True


def _collect_classes_from_module(package_root):
    results = []
    for py in sorted(package_root.glob("*.py")):
        if py.name == "__init__.py":
            continue
        rel = py.relative_to(VLA_ROOT).with_suffix("")
        mod_path = ".".join(rel.parts)
        try:
            m = importlib.import_module(mod_path)
        except Exception as e:
            print(f"  [SKIP] import {mod_path} failed: {type(e).__name__}: {e}")
            continue
        for name, obj in inspect.getmembers(m, inspect.isclass):
            if obj.__module__ != mod_path:
                continue
            if _is_concrete_module(obj):
                results.append((obj, mod_path))
    return results


def _shape_of(t):
    if isinstance(t, torch.Tensor):
        return tuple(t.shape)
    if isinstance(t, dict):
        return {k: tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__ for k, v in t.items()}
    return type(t).__name__


def _make_shape_meta():
    return {
        "obs": {
            "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
            "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
            "robot0_eef_pos": {"shape": [3]},
            "robot0_eef_quat": {"shape": [4]},
            "robot0_gripper_qpos": {"shape": [2]},
        },
        "action": {"shape": [7]},
    }


def _make_mock_obs(B=2, n_obs_steps=1):
    """Adapter forward 输入:符合 PADP padp_policy.py 期待的"已经 flatten n_obs_steps 维"格式
    (B*n_obs_steps, C, H, W) 用于 rgb,(B*n_obs_steps, D) 用于 state。"""
    BN = B * n_obs_steps
    return {
        "agentview_image": torch.randn(BN, 3, 84, 84),
        "robot0_eye_in_hand_image": torch.randn(BN, 3, 84, 84),
        "robot0_eef_pos": torch.randn(BN, 3),
        "robot0_eef_quat": torch.randn(BN, 4),
        "robot0_gripper_qpos": torch.randn(BN, 2),
    }


def _try_instantiate_and_forward(cls, mod_path):
    info = {"module": mod_path, "class": cls.__name__}

    # 跳过抽象基类
    if inspect.isabstract(cls):
        info["skipped_reason"] = "abstract base class"
        return False, info

    # 跳过非 nn.Module 或非 BaseAdapter 子类
    if not (inspect.isclass(cls) and issubclass(cls, nn.Module) and issubclass(cls, BaseAdapter)):
        info["skipped_reason"] = "not a BaseAdapter concrete subclass"
        return False, info

    try:
        # 所有具体 Adapter(PADP/DP)接受 shape_meta + vm_encoder + am_encoder + proj_dim
        # mock 一个最小单图 VM 编码器(用 global avg pool 让 output dim 不依赖 input shape)
        class _MockVM(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 32)
            def forward(self, x):
                # x: (B, C, H, W) → global avg pool → (B, 8) → proj → (B, 32)
                B = x.shape[0]
                feat = x.mean(dim=(2, 3))                  # (B, C)
                # 强行把 C 投影到 8 维(用 linear 8←C,如果 C != 8 用 avg over channels)
                if feat.shape[-1] != 8:
                    feat = feat.mean(dim=-1, keepdim=True).expand(-1, 8)
                return self.proj(feat)
            def output_shape(self):
                return (32,)
        vm = _MockVM()
        kwargs = dict(
            shape_meta=_make_shape_meta(),
            vm_encoder=vm,
            am_encoder=None,
            proj_dim=64,
        )
        m = cls(**kwargs)
        obs = _make_mock_obs(B=2, n_obs_steps=1)
        m.eval()
        with torch.no_grad():
            out = m(obs)
        info["params"] = list(inspect.signature(cls.__init__).parameters.keys())
        info["vm_class"] = type(m.vm).__name__ if hasattr(m, "vm") else None
        info["am_class"] = type(m.am).__name__ if hasattr(m, "am") else None
        info["input_obs_keys"] = list(obs.keys())
        info["input_obs_shapes"] = {k: tuple(v.shape) for k, v in obs.items()}
        info["output"] = _shape_of(out)
        info["params_count"] = sum(p.numel() for p in m.parameters())
        if hasattr(m, "vm") and hasattr(m.vm, "output_shape"):
            info["vm_output_shape"] = tuple(m.vm.output_shape())
        return True, info

    except Exception as e:
        import traceback
        tb = traceback.format_exc().strip().split("\n")[-3:]
        info["error"] = f"{type(e).__name__}: {e}  ||  {' || '.join(tb)}"
        return False, info


def main():
    print("=" * 80)
    print(f"B_model/test_adapters_output.py")
    print(f"扫描目录: B_model/adapters/")
    print("=" * 80)

    pkg_path = THIS_DIR / "adapters"
    if not pkg_path.exists():
        print(f"ERROR: {pkg_path} not found")
        return 1

    print(f"\n[adapters] scanning {pkg_path}")

    classes = _collect_classes_from_module(pkg_path)
    total_ok, total_fail = 0, 0

    for cls, mod_path in classes:
        ok, info = _try_instantiate_and_forward(cls, mod_path)
        tag = "OK " if ok else "FAIL"
        if ok:
            total_ok += 1
        else:
            total_fail += 1
        print(f"  [{tag}] {info['class']:30s} ({mod_path})")
        if ok:
            print(f"        params: {info.get('params_count', '?'):>12,}")
            print(f"        input_obs_keys: {info['input_obs_keys']}")
            for k, v in info["input_obs_shapes"].items():
                print(f"          - {k}: {v}")
            print(f"        output (global_cond): {info['output']}")
            if "vm_output_shape" in info:
                print(f"        vm.output_shape(): {info['vm_output_shape']}")
        else:
            print(f"        error: {info.get('error', 'unknown')}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {total_ok} ok, {total_fail} fail")
    print("=" * 80)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())