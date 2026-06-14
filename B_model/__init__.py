# -*- coding: utf-8 -*-
"""B_model —— 纯算力层(encoders / networks / adapters)。

v5 铁律 2:B_model 不定义 Policy 类,不知道策略大脑的存在。
Gpolicy/<algo>/<algo>_policy.py 在 __init__ 里 import B_model 组件并组装成 Fat Policy。
本文件只导出 B_model 子包,不触发任何 Policy 注册(Policy 在 Gpolicy 里)。
"""
# 触发子包 __init__.py 的 import 副作用(VM/DM 实例化准备)
from B_model import encoders   # noqa: F401
from B_model import networks   # noqa: F401
from B_model import adapters   # noqa: F401