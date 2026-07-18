# CoMER: 基于覆盖机制的 Transformer 手写数学公式识别

[![arXiv](https://img.shields.io/badge/arXiv-2207.04410-b31b1b.svg)](https://arxiv.org/abs/2207.04410)

## 项目结构（集群路径）

```
/home/scc/pb23050866/
├── comer/                     # 模型定义代码包
├── convert2symLG/             # LaTeX 转 symLG 格式工具
├── lgeval/                    # symLG 对比评估工具（CROHME 官方）
├── config.yaml                # CoMER 超参数配置文件
├── config.yaml.bak
├── data/
│   └── data/                  # CROHME 训练数据（2014, 2016, 2019, train）
├── data.zip                   # 数据压缩包
├── eval_all.sh                # 全 CROHME 测试集评估脚本
├── example/
│   ├── UN19_1041_em_595.bmp
│   └── example.ipynb          # HMER 演示
├── lightning_logs/            # 训练日志 & checkpoints
│   ├── version_0/
│   ├── version_23528/
│   └── version_23573/
├── logs/                      # sbatch 作业日志（stdout/stderr）
├── material/                  # 参考资料
├── scripts/                   # 评估脚本
├── requirements.txt
├── setup.cfg
├── setup.py
├── train.py
├── run_comer.sbatch           # Slurm 作业提交脚本
├── run_comer2.sbatch
├── comer_p107.sbatch          # P107-RTX5090 分区专用脚本
├── test_gpu.sbatch
└── .gitignore
```

## 安装依赖
环境针对本科生算力平台的RTX5090，建议创建vscode应用。如果之前没有申请107杯，应用最长时间只能是12小时。配置如下：
![Uploading image.png…]()



```bash
提供打包好的环境，方便使用
链接：https://pan.ustc.edu.cn/share/index/15c255875b1647c9975e?p=1
密码：tf7q
有效期：2026-08-15 23:59:59

得到.tar.gz文件后，类似于
mkdir -p ~/miniconda3/envs/comer_env
tar -xzf comer_env.tar.gz -C ~/miniconda3/envs/comer_env
conda activate comer_env
conda-unpack

具体平台上可能有调整，可以问问大模型
```

## 训练

```bash
见后面
```

在 4 张 NVIDIA 2080Ti 上训练约需 7-8 小时（ddp）。

### 长时间训练（后台运行）

```bash
nohup torchrun --nproc_per_node=4 train.py fit --config config.yaml > train.log 2>&1 &
```

### 测试时（前台运行）

```bash
torchrun --nproc_per_node=4 train.py fit --config config.yaml
```

### 多组训练

同时进行多个训练时，建议自定义日志文件名，避免混淆：

```bash
nohup torchrun --nproc_per_node=4 train.py fit --config config.yaml > train_experiment1.log 2>&1 &
nohup torchrun --nproc_per_node=4 train.py fit --config config.yaml > train_experiment2.log 2>&1 &
```

### 查看训练日志

```bash
tail -f train.log
# 或
tail -f train_experiment1.log
```

### 注意事项

- `torchrun --nproc_per_node=4` 表示使用 4 张 GPU 进行分布式训练
- 如果是单卡训练，去掉 `--nproc_per_node` 参数直接用 `python train.py fit --config config.yaml`
- 如需调整 GPU 数量，修改 `--nproc_per_node` 的值即可

## 调整参数（显存/CPU 瓶颈）

见 `config.yaml` 约第 102 行：

```yaml
train_batch_size: 16    # 显存不够时减小
eval_batch_size: 1
num_workers: 2           # CPU 瓶颈时减小
```

精度调整，`config.yaml` 约第 62 行：

```yaml
precision: 'bf16'   # 将 32 改为 bf16 可显著减少显存占用
```

`torch.set_float32_matmul_precision('medium')`，位于 `train.py` 约第 7 行。

图片尺寸过滤，`comer/datamodule/datamodule.py` 约第 18 行：

```python
MAX_SIZE = 1.6e5  # 超过此像素数的图片会被跳过
```

过大图片会挤占显存，可酌情调整此阈值。

## 评估

使用 CROHME 2019 官方工具进行评估：

```bash
perl --version  # 需要 Perl 5
unzip -q data.zip
bash eval_all.sh 0
```

预训练的 CoMER(Fusion) checkpoint 位于 `lightning_logs/version_0/`。

## Checkpoints

checkpoints 保存在 `/home/scc/pb23050866/lightning_logs/`。用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir lightning_logs
```

每个 checkpoint 约 50MB，训练约 150 个 epoch 即可收敛。但是代码设置300epoch才停止

由于体积较大，checkpoints 未包含在此仓库中，分享方式再议

## 数据

- CROHME 2014/2016/2019 数据集位于 `/home/scc/pb23050866/data/data/`
- 压缩包 `data.zip` 也在项目根目录
- 数据同样未包含在仓库中，需单独获取
