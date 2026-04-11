#!/bin/bash
# Run this script ONCE on the login node (which has internet access)
# to pre-download all HuggingFace assets before submitting SLURM jobs.
#
# Usage:
#   bash download_assets.sh

set -euo pipefail

module purge
module load GCCcore/13.3.0
module load Python/3.12.3

source /scratch/user/vinaysingh/cb-venv/bin/activate

cd /scratch/user/vinaysingh/Cognitive-Bridge

export HF_HOME=/scratch/user/vinaysingh/.cache/huggingface
export PIP_CACHE_DIR=/scratch/user/vinaysingh/PIP_CACHE
mkdir -p "$HF_HOME"

# ─── NER assets ───────────────────────────────────────────────────────────────

echo "=== NER Step 1: Prepare BC5CDR dataset ==="
python3 training/data/prepare_bc5cdr.py

echo ""
echo "=== NER Step 2: Pre-download DistilBioBERT model & tokenizer ==="
python3 -c "
from transformers import AutoTokenizer, AutoModelForTokenClassification
model_name = 'nlpie/distil-biobert'
print(f'Downloading tokenizer: {model_name}')
AutoTokenizer.from_pretrained(model_name)
print(f'Downloading model: {model_name}')
AutoModelForTokenClassification.from_pretrained(model_name)
print('Model & tokenizer cached.')
"

echo ""
echo "=== NER Step 3: Pre-download seqeval metric ==="
python3 -c "
import evaluate
print('Downloading seqeval metric...')
evaluate.load('seqeval')
print('seqeval metric cached.')
"

# ─── T5 simplification assets ────────────────────────────────────────────────

echo ""
echo "=== T5 Step 1: Prepare simplification dataset (Cochrane + curated) ==="
python3 training/data/prepare_simplification.py

echo ""
echo "=== T5 Step 2: Pre-download T5-small model & tokenizer ==="
python3 -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = 'google/flan-t5-small'
print(f'Downloading tokenizer: {model_name}')
AutoTokenizer.from_pretrained(model_name)
print(f'Downloading model: {model_name}')
AutoModelForSeq2SeqLM.from_pretrained(model_name)
print('Model & tokenizer cached.')
"

echo ""
echo "=== T5 Step 3: Pre-download ROUGE metric ==="
python3 -c "
import evaluate
print('Downloading rouge metric...')
evaluate.load('rouge')
print('rouge metric cached.')
"

echo ""
echo "=== T5 Step 4: Pre-download NLTK punkt tokenizer ==="
python3 -c "
import nltk, os
nltk_dir = os.path.join(os.getcwd(), '.nltk_data')
os.makedirs(nltk_dir, exist_ok=True)
nltk.download('punkt', download_dir=nltk_dir, quiet=False)
nltk.download('punkt_tab', download_dir=nltk_dir, quiet=False)
print(f'NLTK data saved to {nltk_dir}')
"

echo ""
echo "=== All assets downloaded. You can now submit SLURM jobs: ==="
echo "    sbatch train_ner.slurm"
echo "    sbatch train_t5.slurm"
