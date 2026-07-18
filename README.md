# CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition

[![arXiv](https://img.shields.io/badge/arXiv-2207.04410-b31b1b.svg)](https://arxiv.org/abs/2207.04410)

## Project structure

```
scc/  (集群根目录 /home/scc/pb23050866/)
├── comer/                     # model definition folder
├── convert2symLG/             # official tool to convert latex to symLG format
├── lgeval/                    # official tool to compare symLGs in two folder
├── config.yaml                # config for CoMER hyperparameters
├── config.yaml.bak
├── data/
│   └── data/                  # CROHME training data (2014, 2016, 2019, train)
├── data.zip                   # compressed data archive
├── eval_all.sh                # script to evaluate model on all CROHME test sets
├── example/
│   ├── UN19_1041_em_595.bmp
│   └── example.ipynb          # HMER demo
├── lightning_logs/            # training logs
│   ├── version_0/
│   ├── version_23528/
│   └── version_23573/
├── logs/                      # sbatch job logs (stdout/stderr)
├── material/                  # reference materials
├── scripts/                   # evaluation scripts
├── requirements.txt
├── setup.cfg
├── setup.py
├── train.py
├── run_comer.sbatch           # sbatch job script for Slurm
├── run_comer2.sbatch          # updated sbatch job script
├── comer_p107.sbatch          # P107-RTX5090 partition sbatch
├── test_gpu.sbatch
└── .gitignore
```

## Install dependencies

```bash
cd /home/scc/pb23050866

# create conda environment
conda create -y -n comer python=3.7
conda activate comer

# install PyTorch
conda install pytorch=1.8.1 torchvision=0.2.2 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia

# training dependencies
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge

# evaluation dependencies
conda install pandoc=1.19.2.1 -c conda-forge

# install project
pip install -e .
```

## Training

Navigate to project root and run:

```bash
python train.py --config config.yaml
```

Training takes ~7-8 hours on 4x NVIDIA 2080Ti GPUs using ddp.

Set different models by editing `config.yaml`:

- **BTTR (baseline)**: `cross_coverage: false, self_coverage: false`
- **CoMER(Self)**: `cross_coverage: false, self_coverage: true`
- **CoMER(Cross)**: `cross_coverage: true, self_coverage: false`
- **CoMER(Fusion)**: `cross_coverage: true, self_coverage: true`

For single GPU, set `gpus: 1` in config.yaml.

## Training on USTC 107 Cluster (Slurm)

The cluster uses Slurm job scheduler. Submit training jobs via sbatch:

```bash
sbatch run_comer.sbatch
```

Available partitions:
- **P107-RTX5090** (RTX 5090, Blackwell sm_120)
- **P107-A100** (A100 80GB)

See the sbatch scripts for configuration details (`comer_p107.sbatch`, `run_comer.sbatch`).

**⚠️ Important notes:**
- PyTorch 2.10.0+ required for RTX 5090 (sm_120)
- Install with: `pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130 --no-deps`
- The login node (tradmin-02) has no GPU — always submit to compute nodes

## Evaluation

Use the official CROHME 2019 evaluation tools:

```bash
perl --version  # ensure Perl 5 is installed
unzip -q data.zip
bash eval_all.sh 0
```

A trained CoMER(Fusion) checkpoint is saved in `lightning_logs/version_0/`.

## Checkpoints

Checkpoints are stored in `/home/scc/pb23050866/lightning_logs/`. View training curves with TensorBoard:

```bash
tensorboard --logdir lightning_logs
```

## Data

- CROHME 2014/2016/2019 datasets at `/home/scc/pb23050866/data/data/`
- Data is also available as `data.zip` at the project root
- Training converges around ~150 epochs