#!/bin/bash
#SBATCH --job-name=cb-t5-onnx-export
#SBATCH --output=logs/onnx_export_%j.out
#SBATCH --error=logs/onnx_export_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vinaysingh@tamu.edu

set -euo pipefail

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

source /scratch/user/vinaysingh/cb-venv/bin/activate

cd /scratch/user/vinaysingh/Cognitive-Bridge

export HF_HOME=/scratch/user/vinaysingh/.cache/huggingface
export PIP_CACHE_DIR=/scratch/user/vinaysingh/PIP_CACHE
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ORT_INTRA_OP_NUM_THREADS=1
export ORT_INTER_OP_NUM_THREADS=1
export ORT_DISABLE_THREAD_AFFINITY=1
export ORT_LOG_SEVERITY_LEVEL=3

mkdir -p logs

echo "=== Job Info ==="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Python:  $(python3 --version)"
echo "================"

python3 training/simplification/export_onnx_t5_medisimplifier.py

echo ""
echo "=== DONE @ $(date -Is) ==="
echo "Next: sbatch verify_onnx_export.slurm"
