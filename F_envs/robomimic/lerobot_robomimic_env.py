# ============================================================
# PADP-VLA v1.3
# 1.3 版本,新加 LerobotRobomimicEnv 包装类 (Method C 核心)
# ============================================================

"""F_envs.robomimic.lerobot_robomimic_env —— LerobotRobomimicEnv (v1.3 包装类)。

==========================================================================
v1.3 Method C 核心:零修改 RobomimicEnv,新加包装类
==========================================================================

本类**包装** `RobomimicEnv`,不修改它(用户红线:padp_for_test 线 1:1 不动)。

核心能力:`obs_to_lerobot()` 把 robomimic 原生 obs 转 lerobot 标准 obs:
  - agentview_image (C, H, W) uint8       → observation.image       (C, H, W) uint8
  - robot0_eye_in_hand_image             → observation.wrist_image
  - robot0_eef_pos (3,)                  ↘
  - robot0_eef_quat (4,) wxyz            → observation.state        (8,)  [pos3 + quat_xyzw + gripper1]
  - robot0_gripper_qpos[:, 0] (1,)       ↗

用 lerobot 数据集训出来的 policy 在 sim rollout 时:
  - raw_env (RobomimicEnv) 跑 robomimic sim
  - obs_to_lerobot() 把 robomimic obs 转 lerobot 标准 obs
  - server 端用 lerobot-trained ckpt,policy 看到的是 lerobot 标准 obs

调用方:
  - `tcp_rollout_client` 创建 `LerobotRobomimicEnv(RobomimicEnv(...))` 而非 `RobomimicEnv(...)`
  - 其他逻辑 (server 通信、action 7↔10 转换) 跟 1.1 完全一样

vs 1.1 (RobomimicEnv):
  + 多了 obs_to_lerobot() 转换
  + step/reset 返回 lerobot 风格 obs
  - 训练时 obs 也是 lerobot 风格(已经由 convert_to_lerobot.py 转好)
  - 所以 sim rollout 跟训练数据 obs schema 一致
==========================================================================
"""
from __future__ import annotations

from typing import Optional, Union, Dict, Any

import numpy as np


class LerobotRobomimicEnv:
    """v1.3 包装类:RobomimicEnv + obs_to_lerobot() 转换。

    包装而非继承,确保 RobomimicEnv 一行不改。
    """

    def __init__(self, raw_env):
        """
        Args:
            raw_env: 已经实例化的 `RobomimicEnv` 对象(由调用方负责创建)
        """
        self.raw_env = raw_env
        # 继承 raw_env 的常用属性(让调用方能透明使用)
        self.dataset_path = raw_env.dataset_path
        self.shape_meta = raw_env.shape_meta
        self.action_space = raw_env.action_space
        self.observation_space = raw_env.observation_space
        self.max_steps = raw_env.max_steps

    # ============================================================
    # 生命周期:转发到 raw_env,加 obs_to_lerobot 转换
    # ============================================================
    def reset(self, seed: Optional[int] = None) -> dict:
        """重置 env,返回 lerobot 风格 obs dict。"""
        raw_obs = self.raw_env.reset(seed=seed)
        return self.obs_to_lerobot(raw_obs)

    def reset_to(self, init_state: Union[np.ndarray, dict]) -> dict:
        """重置到指定 state,返回 lerobot 风格 obs dict。"""
        raw_obs = self.raw_env.reset_to(init_state)
        return self.obs_to_lerobot(raw_obs)

    def seed(self, seed: int) -> None:
        self.raw_env.seed(seed)

    def step(self, action: np.ndarray):
        """执行一步,返回 (obs_lerobot, reward, done, info)。"""
        raw_obs, reward, done, info = self.raw_env.step(action)
        return self.obs_to_lerobot(raw_obs), float(reward), bool(done), dict(info) if info else {}

    def close(self) -> None:
        self.raw_env.close()

    def get_robot_state(self) -> np.ndarray:
        return self.raw_env.get_robot_state()

    def __str__(self) -> str:
        return f"LerobotRobomimicEnv(raw={self.raw_env})"

    # ============================================================
    # v1.3 核心:obs_to_lerobot
    # ============================================================
    def obs_to_lerobot(self, obs: dict) -> dict:
        """把 robomimic 原生 obs dict 转 lerobot 标准 obs dict。

        输入约定 (raw RobomimicEnv 返回):
          - agentview_image:           (C, H, W) uint8
          - robot0_eye_in_hand_image:   (C, H, W) uint8
          - robot0_eef_pos:             (3,)  float32
          - robot0_eef_quat:            (4,)  float32, wxyz 顺序
          - robot0_gripper_qpos:        (2,)  float32, 只取 [:, 0]

        输出 (lerobot 标准):
          - observation.image:       (C, H, W) uint8
          - observation.wrist_image: (C, H, W) uint8
          - observation.state:       (8,)   float32  [pos3 + quat_xyzw + gripper1]

        注意:
          - 转换是确定性的(无随机),单测时 round-trip 不会完全相等(因为 lerobot
            路径是 normalized 后的数据),但 shape / dtype / range 应当跟 lerobot
            parquet 一致
          - quat 顺序:robomimic 存 wxyz,lerobot 期望 xyzw → [1, 2, 3, 0]
          - gripper shape:robomimic 存 (T, 2),lerobot 期望 (T, 1) → [:, :1]
        """
        out: Dict[str, np.ndarray] = {}

        # === RGB keys (零拷贝,只换名字) ===
        if "agentview_image" in obs:
            out["observation.image"] = obs["agentview_image"]
        if "robot0_eye_in_hand_image" in obs:
            out["observation.wrist_image"] = obs["robot0_eye_in_hand_image"]

        # === 8D state:pos3 + quat_xyzw + gripper1 ===
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3, dtype=np.float32))
        eef_quat_wxyz = obs.get("robot0_eef_quat", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        gripper = obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32))

        # 防御 shape:确保是 1D
        eef_pos = np.asarray(eef_pos, dtype=np.float32).flatten()[:3]
        eef_quat_wxyz = np.asarray(eef_quat_wxyz, dtype=np.float32).flatten()[:4]
        gripper = np.asarray(gripper, dtype=np.float32).flatten()[:1]

        # quat 顺序 wxyz → xyzw
        eef_quat_xyzw = eef_quat_wxyz[[1, 2, 3, 0]]

        state_8d = np.concatenate([eef_pos, eef_quat_xyzw, gripper], axis=-1).astype(np.float32)
        out["observation.state"] = state_8d

        return out


# ============================================================
# 5 项单测 (sub-agent 实施时用)
# ============================================================
def _run_unit_tests():
    """5 项单测,验证 obs_to_lerobot() 正确性。

    用法:python -c "from F_envs.robomimic.lerobot_robomimic_env import _run_unit_tests; _run_unit_tests()"
    """
    import unittest

    class TestObsToLerobot(unittest.TestCase):
        def setUp(self):
            self.env = LerobotRobomimicEnv.__new__(LerobotRobomimicEnv)
            self.env.raw_env = None  # 不需要真 env,直接测 obs_to_lerobot

        def test_01_keys(self):
            """测 1:输出 keys 必须是 3 个 lerobot 标准 key"""
            raw = {
                "agentview_image": np.zeros((3, 84, 84), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((3, 84, 84), dtype=np.uint8),
                "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            }
            out = self.env.obs_to_lerobot(raw)
            self.assertEqual(set(out.keys()),
                             {"observation.image", "observation.wrist_image", "observation.state"})

        def test_02_shapes(self):
            """测 2:输出 shapes 必须匹配 lerobot parquet schema"""
            raw = {
                "agentview_image": np.zeros((3, 84, 84), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((3, 84, 84), dtype=np.uint8),
                "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            }
            out = self.env.obs_to_lerobot(raw)
            self.assertEqual(out["observation.image"].shape, (3, 84, 84))
            self.assertEqual(out["observation.wrist_image"].shape, (3, 84, 84))
            self.assertEqual(out["observation.state"].shape, (8,))

        def test_03_dtypes(self):
            """测 3:输出 dtypes 必须匹配 lerobot parquet schema"""
            raw = {
                "agentview_image": np.zeros((3, 84, 84), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((3, 84, 84), dtype=np.uint8),
                "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            }
            out = self.env.obs_to_lerobot(raw)
            self.assertEqual(out["observation.image"].dtype, np.uint8)
            self.assertEqual(out["observation.wrist_image"].dtype, np.uint8)
            self.assertEqual(out["observation.state"].dtype, np.float32)

        def test_04_ranges(self):
            """测 4:quat xyzw 分量必须在 [-1, 1]; pos 任意范围; gripper 通常 [0, 0.04]"""
            raw = {
                "agentview_image": np.zeros((3, 84, 84), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((3, 84, 84), dtype=np.uint8),
                "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "robot0_eef_quat": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),  # wxyz
                "robot0_gripper_qpos": np.array([0.04, 0.0], dtype=np.float32),
            }
            out = self.env.obs_to_lerobot(raw)
            # pos: 直接转发(float32 精度,用 almost_equal)
            np.testing.assert_array_almost_equal(out["observation.state"][:3], [0.1, 0.2, 0.3], decimal=6)
            # quat: wxyz [0.5, 0.5, 0.5, 0.5] → xyzw [0.5, 0.5, 0.5, 0.5] (巧合)
            np.testing.assert_array_almost_equal(out["observation.state"][3:7], [0.5, 0.5, 0.5, 0.5], decimal=6)
            # gripper: 只取 [0]
            self.assertAlmostEqual(out["observation.state"][7], 0.04, places=5)

        def test_05_quat_conversion(self):
            """测 5:quat 顺序 wxyz → xyzw 转换正确性 (单位四元数 [w=1, x=0, y=0, z=0])"""
            raw = {
                "agentview_image": np.zeros((3, 84, 84), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((3, 84, 84), dtype=np.uint8),
                "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # w=1, x=y=z=0
                "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            }
            out = self.env.obs_to_lerobot(raw)
            # xyzw 应为 [x=0, y=0, z=0, w=1]
            np.testing.assert_array_equal(out["observation.state"][3:7], [0.0, 0.0, 0.0, 1.0])
            # 测试 90° 绕 z 轴: wxyz=[cos(45°), 0, 0, sin(45°)]=[0.707, 0, 0, 0.707]
            raw["robot0_eef_quat"] = np.array([0.707, 0.0, 0.0, 0.707], dtype=np.float32)
            out2 = self.env.obs_to_lerobot(raw)
            # xyzw 应为 [x=0, y=0, z=0.707, w=0.707]
            np.testing.assert_allclose(out2["observation.state"][3:7], [0.0, 0.0, 0.707, 0.707], atol=1e-3)

    # 构造 TestSuite 主动跑(避免 __name__ 陷阱)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestObsToLerobot)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    _run_unit_tests()
