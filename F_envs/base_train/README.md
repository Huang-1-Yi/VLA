# F_envs/base_train — padp-vla 训练环境

## 1. 创建环境

```bash
conda env create -f environment.yml
conda activate padp-vla
```

## 2. 验证

```bash
python -c "import torch, einops, zarr, robomimic, gym, diffusers; print('ok')"
```

## 3. 7-Layer 检查

```bash
pytest A_common/tests/ -v
```
