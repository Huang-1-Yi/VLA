# PADP-VLA v2

基于 PADP 的 7-Layer DDD 架构,实现 `square_d0` 单 task 推理成功率 ≥ 90%。

## 快速开始

```bash
# 1. 创建环境(纯 conda,铁律 7)
conda env create -f F_envs/base_train/environment.yml
conda activate padp-vla

# 2. 准备数据
# 把 square_d0_abs.hdf5 放到 data/robomimic/datasets/square_d0/

# 3. 跑契约单元测试
pytest A_common/tests/ -v

# 4. 训练(在 square_d0 上)
python E_cti/train/run_train.py --config E_cti/configs/train_padp.yaml

# 5. 评估
python E_cti/train/run_eval.py \
    --config E_cti/configs/train_padp.yaml \
    --ckpt data/outputs/padp_vla/latest.ckpt \
    --n_test 50
```

## 架构

7 个字母前缀的目录,详见 [执行计划/落地方案v4.md](执行计划/落地方案v4.md)。

## 实施清单

详见 [执行计划/落地方案v4-1的执行清单.md](执行计划/落地方案v4-1的执行清单.md)。
