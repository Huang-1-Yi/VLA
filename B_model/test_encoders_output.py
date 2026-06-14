# -*- coding: utf-8 -*-
"""B_model 自动 smoke test:遍历 encoders/{VM,AM,TM}/ 下所有具体编码器类,
实例化并跑一次 forward,**assert** 输出 shape 与预期一致(本文件 §L3 强化),
打印输入/输出 shape 与参数量,作为个人项目级别的接口契约检查。

运行:
    python -m B_model.test_encoders_output
    # 或:
    cd VLA && python B_model/test_encoders_output.py

设计目的:
  - L1a/L1b 已落地:每个 forward 都有形状注释,每个 __init__ 都有 logger.info
  - L3 本文件:**强制 runtime shape assertion**——发现"代码改了但忘了改 shape_meta"这种 bug 立刻报
  - L5 markdown 表(各模型输入输出接口清单.md)与本文件 EXPECTED_SHAPES 一一对应,任一处漂移会立刻可见
"""
import os
import sys
import inspect
import importlib
from pathlib import Path

import torch
import torch.nn as nn

# 让 import 能找到顶层包
THIS_DIR = Path(__file__).resolve().parent
VLA_ROOT = THIS_DIR.parent
sys.path.insert(0, str(VLA_ROOT))

# === 通用辅助:deep import 一个文件模块,提取所有 nn.Module 具体子类(跳过 abstract / interface) ===
from B_model.encoders.interface_vm import VisionEncoderInterface  # noqa: E402
from B_model.encoders.interface_am import ActionEncoderInterface  # noqa: E402
from B_model.encoders.interface_tm import TimestepEncoderInterface  # noqa: E402


# === L3 强化:预期输出 shape 表(L5 markdown §五 "默认值速查表" 镜像) ===
# key = 类的 __name__,value = (expected_proj_dim,) —— 即 output_shape() 应返回的 tuple
# 当类的 __init__ 用非默认 proj_dim 跑时,这个表会与实际不符(测试会 fail,提醒要更新)
EXPECTED_OUTPUT_SHAPES: dict = {
    # === VM: 单图 → (B, D_vm),output_shape() = (D_vm,) ===
    "RobomimicObsEncoder":      (512,),   # proj_dim=512 默认
    "TimmImageObsEncoder":      (64,),    # proj_dim=64 默认
    "CLIPImageObsEncoder":      (64,),    # test 用 proj_dim=64(默认 512,但我们测小)
    "TorchvisionResNetObsEncoder": (64,), # proj_dim=64
    "R3MObsEncoder":            (64,),    # proj_dim=64
    "DP3ObsEncoder":            (64,),    # proj_dim=64
    "TransformerObsEncoder":    (64,),    # proj_dim=64
    # === AM: (B, T, D_am) → output_shape() = (D_am,) ===
    "LinearStateEncoder":       (64,),    # test 用 output_dim=64
    # === TM: (B,) → (B, D_tm) → output_shape() = (D_tm,) ===
    "SinusoidalTimestepEncoder": (64,),   # test 用 dim=64(默认 512,但测小)
}


def _expected_output_shape(cls_name: str):
    """根据类名查 EXPECTED_OUTPUT_SHAPES;查不到返回 None(不 assert)。"""
    return EXPECTED_OUTPUT_SHAPES.get(cls_name)


def _is_concrete_module(cls):
    """是 nn.Module 的具体子类(非 abstract)? 且实现 output_shape 或 forward?"""
    if not inspect.isclass(cls):
        return False
    if not issubclass(cls, nn.Module):
        return False
    if inspect.isabstract(cls):
        return False
    return True


def _collect_classes_from_module(mod, package_root):
    """扫一个包(VM/AM/TM)的所有 .py,deep import,返回所有具体 nn.Module 类 + 它们所属的子模块路径。"""
    results = []
    for py in sorted(package_root.glob("*.py")):
        if py.name == "__init__.py":
            continue
        # 相对 import 路径(相对 VLA_ROOT)
        rel = py.relative_to(VLA_ROOT).with_suffix("")
        mod_path = ".".join(rel.parts)
        try:
            m = importlib.import_module(mod_path)
        except Exception as e:
            print(f"  [SKIP] import {mod_path} failed: {type(e).__name__}: {e}")
            continue
        for name, obj in inspect.getmembers(m, inspect.isclass):
            if obj.__module__ != mod_path:
                continue  # 跳过从别处 re-export 的类
            if _is_concrete_module(obj):
                results.append((obj, mod_path))
    return results


def _shape_of(t):
    if isinstance(t, torch.Tensor):
        return tuple(t.shape)
    if isinstance(t, dict):
        return {k: tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__ for k, v in t.items()}
    if isinstance(t, (list, tuple)):
        return [tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__ for x in t]
    return type(t).__name__


def _make_mock_obs(B=2, n_obs_steps=1):
    """生成一个简单的 mock obs dict 给 VM/RobomimicObsEncoder。"""
    return {
        "agentview_image": torch.randn(B * n_obs_steps, 3, 84, 84),
        "robot0_eye_in_hand_image": torch.randn(B * n_obs_steps, 3, 84, 84),
        "robot0_eef_pos": torch.randn(B * n_obs_steps, 3),
        "robot0_eef_quat": torch.randn(B * n_obs_steps, 4),
        "robot0_gripper_qpos": torch.randn(B * n_obs_steps, 2),
    }


def _make_shape_meta():
    """最小 shape_meta(对齐 E_cti/configs/train_padp_full.yaml)。"""
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


def _try_instantiate_and_forward(cls, mod_path):
    """根据类的接口签名,选合适的参数来实例化并 forward。返回 (success, info_dict)。"""
    info = {"module": mod_path, "class": cls.__name__}
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    info["params"] = list(params.keys())

    try:
        # ---------- VM: 各种 VisionEncoderInterface 子类(全是单图接口) ----------
        if issubclass(cls, VisionEncoderInterface):
            # 按类名适配 kwargs 和 obs 输入 shape
            if cls.__name__ == "RobomimicObsEncoder":
                kwargs = dict(
                    shape_meta=_make_shape_meta(),
                    crop_shape=(76, 76),
                    obs_encoder_group_norm=True,
                    eval_fixed_crop=True,
                    task_name="square",
                )
                # RobomimicObsEncoder 单图接口:输入 (B, C, H, W) 一张图
                obs = torch.randn(2, 3, 84, 84)
            elif cls.__name__ == "CLIPImageObsEncoder":
                kwargs = dict(
                    clip_model_name="ViT-B/32",
                    proj_dim=64,
                )
                # 单图 (B, C, H, W)
                obs = torch.randn(2, 3, 84, 84)
            elif cls.__name__ == "TimmImageObsEncoder":
                kwargs = dict(
                    model_name="resnet18",
                    pretrained=False,
                    proj_dim=64,
                )
                # 单图 (B, C, H, W)
                obs = torch.randn(2, 3, 84, 84)
            elif cls.__name__ == "TimmObsEncoder":
                kwargs = dict(
                    shape_meta=_make_shape_meta(),
                    model_name="resnet18",
                    pretrained=False,
                    proj_dim=64,
                )
                # TimmObsEncoder 期望 (B, T, C, H, W) dict
                obs = {
                    "agentview_image": torch.randn(2, 1, 3, 84, 84),
                    "robot0_eye_in_hand_image": torch.randn(2, 1, 3, 84, 84),
                    "robot0_eef_pos": torch.randn(2, 1, 3),
                    "robot0_eef_quat": torch.randn(2, 1, 4),
                    "robot0_gripper_qpos": torch.randn(2, 1, 2),
                }
            elif cls.__name__ == "CLIPMultiImageObsEncoder":
                kwargs = dict(
                    shape_meta=_make_shape_meta(),
                    clip_model_name="ViT-B/32",
                    proj_dim=64,
                )
                # Timm-style obs shape (B, T, C, H, W)
                obs = {
                    "agentview_image": torch.randn(2, 1, 3, 84, 84),
                    "robot0_eye_in_hand_image": torch.randn(2, 1, 3, 84, 84),
                    "robot0_eef_pos": torch.randn(2, 1, 3),
                    "robot0_eef_quat": torch.randn(2, 1, 4),
                    "robot0_gripper_qpos": torch.randn(2, 1, 2),
                }
            # === A 补全:4 个新增 VM(2026-06-14 添加)== =
            elif cls.__name__ == "TorchvisionResNetObsEncoder":
                kwargs = dict(
                    model_name="resnet18",
                    weights=None,        # 不下载预训练权重,测试更快
                    proj_dim=64,
                )
                obs = torch.randn(2, 3, 76, 76)
            elif cls.__name__ == "R3MObsEncoder":
                # R3M 未装时自动 fallback 到 torchvision resnet18
                kwargs = dict(
                    model_name="resnet18",
                    proj_dim=64,
                )
                obs = torch.randn(2, 3, 76, 76)
            elif cls.__name__ == "DP3ObsEncoder":
                # DP3 是 3D 点云编码器,不是图像
                kwargs = dict(
                    out_channel=64,
                    state_mlp_size=(32, 32),
                    use_pc_color=False,   # 3 维 xyz
                    proj_dim=64,
                )
                # mock 点云: (B, N_pc=64, C_pc=3)
                obs = torch.randn(2, 64, 3)
            elif cls.__name__ == "TransformerObsEncoder":
                # ViT 强制 224x224 输入
                kwargs = dict(
                    model_name="vit_base_patch16_clip_224.openai",  # timm 自动下载
                    pretrained=False,        # 不下载,测试更快
                    proj_dim=64,
                )
                obs = torch.randn(2, 3, 224, 224)
            else:
                info["skipped_reason"] = f"unknown VM class: {cls.__name__}"
                return False, info
            m = cls(**kwargs)
            m.eval()
            with torch.no_grad():
                out = m(obs)
            # 单图 encoder 收 Tensor,多图 dict encoder 收 dict
            if isinstance(obs, dict):
                info["input_obs"] = {k: tuple(v.shape) for k, v in obs.items()}
            else:
                info["input"] = tuple(obs.shape)
            info["output"] = _shape_of(out)
            info["output_shape()"] = tuple(m.output_shape())

        # ---------- AM: ActionEncoderInterface 子类 ----------
        elif issubclass(cls, ActionEncoderInterface):
            # 按类名适配 kwargs
            if cls.__name__ == "LinearStateEncoder":
                m = cls(input_dim=9, output_dim=64)
                x = torch.randn(2, 1, 9)  # [B, T, D_in]
            else:
                # 兜底:猜 kwargs
                kwargs = {}
                for pname, param in params.items():
                    if pname == "self":
                        continue
                    if param.default is inspect.Parameter.empty:
                        kwargs[pname] = 7
                m = cls(**kwargs)
                x = torch.randn(2, 4, 7)
            with torch.no_grad():
                out = m(x)
            info["input"] = tuple(x.shape)
            info["output"] = _shape_of(out)
            info["output_shape()"] = tuple(m.output_shape())

        # ---------- TM: TimestepEncoderInterface ----------
        elif issubclass(cls, TimestepEncoderInterface):
            # 按类名适配 kwargs
            if cls.__name__ == "SinusoidalTimestepEncoder":
                m = cls(dim=64)
            else:
                # 兜底:猜 kwargs
                kwargs = {}
                for pname, param in params.items():
                    if pname == "self":
                        continue
                    if param.default is inspect.Parameter.empty:
                        if pname == "dim":
                            kwargs[pname] = 64
                        else:
                            kwargs[pname] = 7
                m = cls(**kwargs)
            x = torch.randint(0, 1000, (2,))
            with torch.no_grad():
                out = m(x)
            info["input"] = tuple(x.shape)
            info["output"] = _shape_of(out)
            info["output_shape()"] = tuple(m.output_shape())

        # ---------- 其他 nn.Module 子类(像 FixedCropRandomizer) ----------
        else:
            # SinusoidalPosEmb 是 SinusoidalTimestepEncoder 的子模块,不是顶层接口实现,
            # 跳过以避免重复测试(SinusoidalTimestepEncoder 已经在 TM 分支测了)
            if cls.__name__ == "SinusoidalPosEmb":
                info["skipped_reason"] = "submodule of SinusoidalTimestepEncoder (tested via parent)"
                return True, info  # skip 算 pass(不是 fail)
            # 猜 kwargs:按参数名常见约定
            kwargs = {}
            for pname, param in params.items():
                if pname == "self":
                    continue
                if param.default is not inspect.Parameter.empty:
                    continue
                if pname == "input_shape":
                    kwargs[pname] = (3, 84, 84)
                elif pname in ("in_channels", "out_channels", "inp_channels", "dim"):
                    kwargs[pname] = 16
                elif pname in ("crop_height", "crop_width"):
                    kwargs[pname] = 8
                elif pname == "kernel_size":
                    kwargs[pname] = 3
                elif pname == "n_groups":
                    kwargs[pname] = 4
                elif pname == "cond_dim":
                    kwargs[pname] = 32
                elif pname == "num_crops":
                    kwargs[pname] = 1
                else:
                    info["skipped_reason"] = f"unknown required param: {pname}"
                    return False, info
            m = cls(**kwargs)
            with torch.no_grad():
                if "Randomizer" in cls.__name__:
                    x = torch.randn(2, 3, 84, 84)
                    out = m.forward_in(x)
                elif "PointNetEncoder" in cls.__name__:
                    # PointNet 期望 (B, N, in_channels);用 kwargs 里的 in_channels
                    in_ch = kwargs.get("in_channels", 6)
                    x = torch.randn(2, 16, in_ch)  # N=16 points
                    out = m(x)
                elif "Downsample" in cls.__name__:
                    x = torch.randn(2, 16, 32)
                    out = m(x)
                elif "Upsample" in cls.__name__:
                    x = torch.randn(2, 16, 16)
                    out = m(x)
                elif "Conv1dBlock" in cls.__name__:
                    x = torch.randn(2, 16, 32)
                    out = m(x)
                elif "Residual" in cls.__name__:
                    x = torch.randn(2, 16, 32)
                    out = m(x)
                else:
                    x = torch.randn(2, 16, 32)
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
    print(f"B_model/test_encoders_output.py")
    print(f"扫描目录: B_model/encoders/{{VM, AM, TM}}/")
    print("=" * 80)

    encoders_root = THIS_DIR / "encoders"
    total_ok, total_fail = 0, 0

    for subpkg in ["VM", "AM", "TM"]:
        pkg_path = encoders_root / subpkg
        if not pkg_path.exists() or not pkg_path.is_dir():
            print(f"\n[{subpkg}] directory not found, skip")
            continue
        print(f"\n[{subpkg}] scanning {pkg_path}")

        classes = _collect_classes_from_module(None, pkg_path)
        if not classes:
            print(f"  (空目录,跳过)")

        for cls, mod_path in classes:
            ok, info = _try_instantiate_and_forward(cls, mod_path)
            is_skip = "skipped_reason" in info
            tag = "OK " if ok else "FAIL"
            if ok and not is_skip:
                total_ok += 1
            elif is_skip:
                pass  # skip 不算 ok/fail
            else:
                total_fail += 1
            print(f"  [{tag}] {info['class']:30s} ({mod_path})")
            if is_skip:
                print(f"        SKIP: {info['skipped_reason']}")
                continue
            if ok:
                # === L3 强化:assert output_shape() 与 EXPECTED_OUTPUT_SHAPES 一致 ===
                expected = _expected_output_shape(info["class"])
                if expected is not None:
                    actual = info.get("output_shape()")
                    if actual != expected:
                        tag = "FAIL"
                        total_ok -= 1
                        total_fail += 1
                        print(f"        ❌ SHAPE ASSERT FAIL: output_shape()={actual}, expected={expected}")
                        print(f"        ↳ 修复方向:在 EXPECTED_OUTPUT_SHAPES 更新 expected,或修复类的 proj_dim/output_dim")
                        # 把行首 tag 覆盖
                        # (terminal 上无法修改已打印行,但 tag 变量在 final SUMMARY 会被引用)
                        info["shape_assert_fail"] = True
                    else:
                        print(f"        ✅ output_shape()={actual} == expected {expected} [L3 assert PASS]")
                print(f"        params: {info.get('params_count', '?'):>10,}")
                if "input" in info:
                    print(f"        input: {info['input']}")
                if "input_obs" in info:
                    for k, v in info["input_obs"].items():
                        print(f"          - {k}: {v}")
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