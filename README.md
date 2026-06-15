# PADP-VLA v1.0

> **1.0 版本,可训练 PADP 但无 rollout**

基于 PADP 算法的 VLA (Vision-Language-Action) 训练框架。
在 `square_d0` 任务上 1:1 复刻 PADP_v3 黄金命令训练效果。

## 训练

### 1. 启动训练

```bash
cd ~/Desktop/PADP_v3/VLA
conda activate equidiff
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/equidiff/bin/python E_cti/train/padp_for_test_train.py
```

训练配置(原 `padp_for_test_golden.yaml`)已嵌入训练脚本,无需外部 yaml。
若需用外部 yaml 覆盖,可加 `--config path/to/config.yaml`。

### 2. 烟雾测试(快速跑通 pipeline)

```bash
cd ~/Desktop/PADP_v3/VLA
conda activate equidiff
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/equidiff/bin/python E_cti/train/padp_for_test_train.py --max_epochs 2
```

### 3. 数据集

训练数据为 `data/robomimic/datasets/square_d0/square_d0_abs.hdf5` (robomimic 演示数据,
200 个 demo,hdf5 内部 action 维度为 7,Dataset 内部做 7→10 转换)。

### 4. 训练输出

- 最新 ckpt: `data/outputs/padp_for_test_golden/latest.ckpt`
- Top-5 ckpt (按 train_loss 升序): `data/outputs/padp_for_test_golden/topk/`
- Normalizer: `data/outputs/padp_for_test_golden/normalizer.pt`

支持断点续训:把 `train.resume: true` 配合 `latest.ckpt` 即可。

## 关键训练参数(嵌入默认配置)

| 项 | 值 | 说明 |
|----|----|----|
| num_epochs | 251 | 源端 50000/200+1 |
| batch_size | 64 | 黄金标准 |
| num_workers | 32 | 黄金标准 |
| lr | 1.0e-4 | AdamW |
| betas | (0.95, 0.999) | 源端 |
| lr_warmup_steps | 500 | cosine + warmup |
| ema.power | 0.75 | 源端 |
| ema.max_value | 0.9999 | |
| balanced_sampler | True | RTV8-aligned per-epoch 平衡采样 |
| skip_rollout | True | v1.0 不跑 rollout |

## 架构

```
VLA/
├── A_common/        # 基础设施层(契约代码,跨阶段稳定)
│   ├── ckpt/        # EMAModel + ckpt util
│   ├── data/        # BaseVLADataset, SequenceSampler, BalancedColumnsSampler, base_collate
│   ├── logger/      # get_logger
│   ├── registry/    # policy_registry
│   └── types/       # Normalizer, ActionOutput
├── B_model/         # 纯算力层
│   ├── adapters/    # BaseAdapter, PADPAdapter
│   ├── encoders/    # VisionEncoderInterface, RobomimicObsEncoder
│   └── networks/    # DiffusionNetworkInterface, Unet1DPadp
├── C_sim/           # 仿真边界
│   └── robomimic/   # RobomimicZarrDatasetPadpForTest, rotation_numpy
├── Gpolicy/         # 策略大脑层(Fat Policy)
│   ├── base_policy_abstract.py
│   └── PADP/        # SlidingWindowDiffusionPolicy
└── E_cti/           # 纯执行流
    └── train/
        └── padp_for_test_train.py   # 训练入口
```
