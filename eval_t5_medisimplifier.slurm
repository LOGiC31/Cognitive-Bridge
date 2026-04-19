#!/bin/bash
#SBATCH --job-name=cb-t5-medisimplifier-eval
#SBATCH --output=logs/t5_medisimplifier_eval_%j.out
#SBATCH --error=logs/t5_medisimplifier_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --ntasks=1

set -euo pipefail

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

# Activate environment (prefer conda env "cb-env")
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cb-env
elif [[ -f /scratch/user/vinaysingh/cb-env/bin/activate ]]; then
  # Alternate: venv located at /scratch/user/vinaysingh/cb-env
  # shellcheck disable=SC1091
  source /scratch/user/vinaysingh/cb-env/bin/activate
else
  echo "WARN: could not activate cb-env; falling back to /scratch/user/vinaysingh/cb-venv" >&2
  # shellcheck disable=SC1091
  source /scratch/user/vinaysingh/cb-venv/bin/activate
fi

cd /scratch/user/vinaysingh/Cognitive-Bridge

export HF_HOME=/scratch/user/vinaysingh/.cache/huggingface
export PIP_CACHE_DIR=/scratch/user/vinaysingh/PIP_CACHE
export NLTK_DATA=/scratch/user/vinaysingh/Cognitive-Bridge/.nltk_data

# Offline mode — ROUGE metric must be pre-cached via download_assets.sh on login node.
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Reduce noisy ORT stderr during ONNX eval under Slurm.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export ORT_INTRA_OP_NUM_THREADS="${ORT_INTRA_OP_NUM_THREADS:-1}"
export ORT_INTER_OP_NUM_THREADS="${ORT_INTER_OP_NUM_THREADS:-1}"
export ORT_DISABLE_THREAD_AFFINITY="${ORT_DISABLE_THREAD_AFFINITY:-1}"
export ORT_LOG_SEVERITY_LEVEL="${ORT_LOG_SEVERITY_LEVEL:-3}"

mkdir -p logs
mkdir -p "$HF_HOME"

echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python:    $(python3 --version)"
echo "Venv:      $(python3 -c 'import sys; print(sys.executable)')"
echo "CUDA:      $(python3 - <<'PY'
import torch
avail = torch.cuda.is_available()
dev = torch.cuda.get_device_name(0) if avail else None
print(f"available={avail}, device={dev}")
PY
)"
echo "HF_HOME:   ${HF_HOME}"
echo "OFFLINE:   HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE} HF_EVALUATE_OFFLINE=${HF_EVALUATE_OFFLINE} HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "================"

# Tune these if you want a faster smoke eval vs a fuller eval.
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_BEAMS="${NUM_BEAMS:-4}"
QUAL_SAMPLES="${QUAL_SAMPLES:-8}"
QUAL_OUT="${QUAL_OUT:-logs/t5_medisimplifier_qual_${SLURM_JOB_ID}.jsonl}"

echo ""
echo "Eval settings:"
echo "  SPLIT=${SPLIT}"
echo "  MAX_SAMPLES=${MAX_SAMPLES}"
echo "  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "  NUM_BEAMS=${NUM_BEAMS}"
echo "  QUAL_SAMPLES=${QUAL_SAMPLES}"
echo "  QUAL_OUT=${QUAL_OUT}"
echo ""

python3 training/simplification/evaluate_t5_medisimplifier.py \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --num_beams "${NUM_BEAMS}" \
  --onnx_provider auto \
  --qualitative_samples "${QUAL_SAMPLES}" \
  --qualitative_out "${QUAL_OUT}"

echo ""
echo "=== DONE @ $(date -Is) ==="
