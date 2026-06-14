# -*- coding: utf-8 -*-
"""B_model 自动 smoke test:遍历 networks/DM/ 下所有具体主干类(包括 UNet 主类 + 内部组件),
实例化并跑一次 forward,打印输出 shape 和参数量。

运行:
    cd VLA && python B_model/test_networks_output.py
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

from B_model.networks.interface_dm import DiffusionNetworkInterface  # noqa: E402


def _is_concrete_module(cls):
    if not inspect.isclass(cls):
        return False
    if not issubclass(cls, nn.Module):
        return False
    if inspect.isabstract(cls):
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


def _try_instantiate_and_forward(cls, mod_path):
    info = {"module": mod_path, "class": cls.__name__}
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    info["init_params"] = list(params.keys())

    try:
        kwargs = {}
        for pname, param in params.items():
            if pname == "self":
                continue
            # 无默认值的必需参数:给个合理 mock 值
            if param.default is inspect.Parameter.empty:
                if pname in ("input_dim", "in_channels", "inp_channels", "dim"):
                    kwargs[pname] = 7
                elif pname == "out_channels":
                    kwargs[pname] = 16
                elif pname == "global_cond_dim":
                    kwargs[pname] = 32
                elif pname == "cond_dim":
                    kwargs[pname] = 16
                elif pname == "kernel_size":
                    kwargs[pname] = 3
                else:
                    info["skipped_reason"] = f"unknown required param: {pname}"
                    return False, info
                continue

            # 有默认值的参数:对 DM/UNet 类强制覆盖(因为 UNet 的默认 global_cond_dim=None 会报错)
            if pname == "input_dim":
                kwargs[pname] = 7
            elif pname == "local_cond_dim":
                kwargs[pname] = None
            elif pname == "global_cond_dim":
                kwargs[pname] = 32
            elif pname == "down_dims":
                kwargs[pname] = (16, 32, 64)
            elif pname == "kernel_size":
                kwargs[pname] = 3
            elif pname == "n_groups":
                kwargs[pname] = 4
            elif pname == "cond_predict_scale":
                kwargs[pname] = False

        m = cls(**kwargs)

        # ---------- DM 主类:Unet1DPadp(DiffusionNetworkInterface) ----------
        if issubclass(cls, DiffusionNetworkInterface):
            input_dim = kwargs.get("input_dim", 7)
            global_cond_dim = kwargs.get("global_cond_dim", 32)
            B, H = 2, 16
            sample = torch.randn(B, H, input_dim)
            global_cond = torch.randn(B, global_cond_dim)
            with torch.no_grad():
                out = m(sample, global_cond=global_cond)
            info["input"] = {"sample": tuple(sample.shape), "global_cond": tuple(global_cond.shape)}
            info["output"] = _shape_of(out)
            info["output_shape()"] = tuple(m.output_shape())

        # ---------- 内部组件:Downsample1d / Upsample1d ----------
        elif "Downsample" in cls.__name__:
            dim = kwargs.get("dim", 16)
            x = torch.randn(2, dim, 32)
            with torch.no_grad():
                out = m(x)
            info["input"] = tuple(x.shape)
            info["output"] = tuple(out.shape)
        elif "Upsample" in cls.__name__:
            dim = kwargs.get("dim", 16)
            x = torch.randn(2, dim, 16)
            with torch.no_grad():
                out = m(x)
            info["input"] = tuple(x.shape)
            info["output"] = tuple(out.shape)

        # ---------- Conv1dBlock ----------
        elif cls.__name__ == "Conv1dBlock":
            inp = kwargs.get("inp_channels", 16)
            out = kwargs.get("out_channels", 16)
            x = torch.randn(2, inp, 32)
            with torch.no_grad():
                y = m(x)
            info["input"] = tuple(x.shape)
            info["output"] = tuple(y.shape)

        # ---------- ConditionalResidualBlock1D ----------
        elif "Residual" in cls.__name__ or "ResBlock" in cls.__name__:
            in_ch = kwargs.get("in_channels", 16)
            out_ch = kwargs.get("out_channels", 16)
            cond = kwargs.get("cond_dim", 16)
            x = torch.randn(2, in_ch, 32)
            cond_t = torch.randn(2, cond)
            with torch.no_grad():
                try:
                    y = m(x, cond_t)
                except TypeError:
                    y = m(x, cond=cond_t)
            info["input"] = {"x": tuple(x.shape), "cond": tuple(cond_t.shape)}
            info["output"] = tuple(y.shape)

        else:
            # 兜底:用 mock 全连接输入试
            x = torch.randn(2, 16, 32)
            with torch.no_grad():
                out = m(x)
            info["input"] = tuple(x.shape)
            info["output"] = _shape_of(out)

        info["params_count"] = sum(p.numel() for p in m.parameters())
        return True, info

    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return False, info


def main():
    print("=" * 80)
    print(f"B_model/test_networks_output.py")
    print(f"扫描目录: B_model/networks/DM/")
    print("=" * 80)

    pkg_path = THIS_DIR / "networks" / "DM"
    if not pkg_path.exists():
        print(f"ERROR: {pkg_path} not found")
        return 1

    print(f"\n[DM] scanning {pkg_path}")

    classes = _collect_classes_from_module(pkg_path)
    total_ok, total_fail = 0, 0

    # 按 "重要性" 排序:主类 Unet1DPadp 放最前
    classes.sort(key=lambda x: (0 if x[0].__name__ == "Unet1DPadp" else 1, x[0].__name__))

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
            if "input" in info and isinstance(info["input"], dict):
                for k, v in info["input"].items():
                    print(f"          - {k}: {v}")
            else:
                print(f"        input: {info.get('input')}")
            print(f"        output: {info['output']}")
            if "output_shape()" in info:
                print(f"        output_shape(): {info['output_shape()']}")
        else:
            print(f"        error: {info.get('error', info.get('skipped_reason', 'unknown'))}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {total_ok} ok, {total_fail} fail")
    print("=" * 80)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())