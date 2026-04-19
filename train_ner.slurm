#!/bin/bash
#SBATCH --job-name=cb-ner
#SBATCH --output=logs/ner_train_%j.out
#SBATCH --error=logs/ner_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --ntasks=1

module purge
module load GCCcore/13.3.0
module load Python/3.12.3
module load nodejs/20.13.1

source /scratch/user/vinaysingh/cb-venv/bin/activate

cd /scratch/user/vinaysingh/Cognitive-Bridge

export HF_HOME=/scratch/user/vinaysingh/.cache/huggingface
export PIP_CACHE_DIR=/scratch/user/vinaysingh/PIP_CACHE

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

echo "Step 1: Preparing BC5CDR dataset..."
python3 training/data/prepare_bc5cdr.py

echo "Step 2: Fine-tuning DistilBERT NER..."
python3 training/ner/finetune_ner.py

echo "Step 3: Exporting to ONNX..."
python3 training/ner/export_onnx_ner.py

echo "=== Done ==="
