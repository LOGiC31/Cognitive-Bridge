#!/bin/bash
#SBATCH --job-name=cb-t5-medisimplifier
#SBATCH --output=logs/t5_medisimplifier_%j.out
#SBATCH --error=logs/t5_medisimplifier_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1           # A40 48GB — A100s currently unavailable; switch back when free
#SBATCH --cpus-per-task=8          # 4 DataLoader workers + headroom
#SBATCH --mem=48G
#SBATCH --time=06:00:00            # ~4-5h expected on A40 (22k pairs, 30 epochs, flan-t5-small)
#SBATCH --ntasks=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vinaysingh@tamu.edu

set -euo pipefail

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

source /scratch/user/vinaysingh/cb-venv/bin/activate

cd /scratch/user/vinaysingh/Cognitive-Bridge

export HF_HOME=/scratch/user/vinaysingh/.cache/huggingface
export PIP_CACHE_DIR=/scratch/user/vinaysingh/PIP_CACHE
export NLTK_DATA=/scratch/user/vinaysingh/Cognitive-Bridge/.nltk_data

# Offline mode — all assets must be pre-cached via download_assets.sh (login node).
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1

# Suppress HuggingFace tokenizer fork warnings (we use multiple DataLoader workers).
export TOKENIZERS_PARALLELISM=false

# ONNX Runtime / OpenMP thread pinning can spam stderr under Slurm cgroup CPU masks.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ORT_INTRA_OP_NUM_THREADS=1
export ORT_INTER_OP_NUM_THREADS=1
export ORT_DISABLE_THREAD_AFFINITY=1
export ORT_LOG_SEVERITY_LEVEL=3     # ERROR only — suppresses noisy quantization warnings

mkdir -p logs
mkdir -p "$HF_HOME"

run_step () {
  local name="$1"
  shift
  echo ""
  echo "===================="
  echo "STEP: ${name}"
  echo "TIME: $(date -Is)"
  echo "CMD : $*"
  echo "===================="
  "$@"
  echo "---- STEP DONE: ${name} @ $(date -Is) ----"
}

echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python:    $(python3 --version)"
echo "Venv:      $(python3 -c 'import sys; print(sys.executable)')"
echo "CUDA:      $(python3 - <<'PY'
import torch
avail = torch.cuda.is_available()
dev   = torch.cuda.get_device_name(0) if avail else None
cap  = torch.cuda.get_device_capability(0) if avail else None
print(f"available={avail}, device={dev}, capability={cap}, bf16={cap is not None and cap[0]>=8}")
PY
)"
echo "HF_HOME:   ${HF_HOME}"
echo "OFFLINE:   HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE} HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "================"

run_step "Prepare dataset (medisimplifier)" \
  python3 training/data/prepare_simplification_medisimplifier.py

run_step "Verify dataset" \
  python3 training/data/verify_simplification_dataset.py \
    --path training/data/simplification_pairs_medisimplifier \
    --tokenizer google/flan-t5-small \
    --max_input_tokens 512 \
    --max_target_tokens 512 \
    --samples 3

run_step "Fine-tune T5 (medisimplifier)" \
  python3 training/simplification/finetune_t5_medisimplifier.py

# Export and quantize disabled — run manually after reviewing training results.
# run_step "Export ONNX + quantize (medisimplifier)" \
#   python3 training/simplification/export_onnx_t5_medisimplifier.py

echo ""
echo "=== DONE @ $(date -Is) ==="

