#!/bin/bash
#SBATCH -J install-torch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 0:30:00
#SBATCH -p P107-RTX5090
#SBATCH -A competition
#SBATCH --qos qos_p107-rtx5090
#SBATCH -o logs/install_torch-%j.out
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate comer2
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall 2>&1 | tail -5
python -c 'import torch; print(f"Torch {torch.__version__}"); print(torch.cuda.get_arch_list())'
echo DONE
