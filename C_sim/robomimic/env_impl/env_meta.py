"""env_meta —— 23 个 robomimic task 的元信息。

源:抄自 PADP `diffusion_policy/common/robomimic_config_util.py`。
"""
from robomimic import get_config


# 23 个 robomimic task 的 max_steps 字典
# 源:抄自 PADP train_sim.py 的 `max_steps` 字典
TASK_MAX_STEPS = {
    "square_d0": 400,
    "square_d1": 400,
    "square_d2": 400,
    "square_d3": 400,
    "can_d0": 400,
    "can_d1": 400,
    "can_d2": 400,
    "can_d3": 400,
    "lift_d0": 400,
    "lift_d1": 400,
    "lift_d2": 400,
    "lift_d3": 400,
    "transport_d0": 700,
    "transport_d1": 700,
    "tool_hang_d0": 700,
    "tool_hang_d1": 700,
    "stack_d0": 500,
    "stack_d1": 500,
    "stack_d2": 500,
    "stack_d3": 500,
    "nut_assembly_d0": 700,
    "nut_assembly_d1": 700,
    "nut_assembly_d2": 700,
    "nut_assembly_d3": 700,
    "pick_place_d0": 400,
    "pick_place_d1": 400,
    "pick_place_d2": 400,
    "pick_place_d3": 400,
    "square": 400,
    "can": 400,
    "lift": 400,
    "transport": 700,
    "tool_hang": 700,
    "stack": 500,
    "nut_assembly": 700,
    "pick_place": 400,
    "kitchen": 1500,
}


def get_max_steps(task_name: str) -> int:
    return TASK_MAX_STEPS.get(task_name, 400)


def get_env_meta(task_name: str = "square", dataset_type: str = "ph"):
    """从 robomimic 拿 EnvUtils config。"""
    config = get_config(
        algo_name="bc_rnn",
        hdf5_type="image",
        task_name=task_name,
        dataset_type=dataset_type,
    )
    return config
