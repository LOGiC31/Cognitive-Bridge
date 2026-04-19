#!/bin/bash
#SBATCH --job-name=cb-t5
#SBATCH --output=logs/t5_train_%j.out
#SBATCH --error=logs/t5_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --ntasks=1

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

source /scratch/user/vinaysingh/cb-venv/bin/activate

cd /scratch/user/vinaysingh/Cognitive-Bridge

export HF_HOME=/scratch/user/vinaysingh/.cache/huggingface
export PIP_CACHE_DIR=/scratch/user/vinaysingh/PIP_CACHE
export NLTK_DATA=/scratch/user/vinaysingh/Cognitive-Bridge/.nltk_data

# Offline mode — all assets must be pre-cached via download_assets.sh
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

mkdir -p logs
mkdir -p "$HF_HOME"

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python: $(python3 --version)"
echo "PyTorch CUDA: $(python3 -c 'import torch; print(f"available={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")')"
echo "================"

set -e

echo "Step 1: Preparing simplification dataset..."
python3 training/data/prepare_simplification.py

echo "Step 2: Fine-tuning T5-small for simplification..."
python3 training/simplification/finetune_t5.py

echo "Step 3: Exporting to ONNX..."
python3 training/simplification/export_onnx_t5.py

echo "=== Done ==="
