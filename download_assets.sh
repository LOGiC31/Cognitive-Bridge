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

# ─── NER assets (already downloaded) ─────────────────────────────────────────

# echo "=== NER Step 1: Prepare BC5CDR dataset ==="
# python3 training/data/prepare_bc5cdr.py

# echo "=== NER Step 2: Pre-download DistilBioBERT model & tokenizer ==="
# python3 -c "
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# AutoTokenizer.from_pretrained('nlpie/distil-biobert')
# AutoModelForTokenClassification.from_pretrained('nlpie/distil-biobert')
# "

# echo "=== NER Step 3: Pre-download seqeval metric ==="
# python3 -c "import evaluate; evaluate.load('seqeval')"

# ─── T5 simplification assets (already downloaded) ───────────────────────────

# echo "=== T5 Step 1: Prepare simplification dataset (Cochrane + curated) ==="
# python3 training/data/prepare_simplification.py

# echo "=== T5 Step 1b: Pre-download medisimplifier dataset ==="
# python3 -c "
# from datasets import load_dataset
# for split in ('train','validation','test'):
#     load_dataset('GuyDor007/medisimplifier-dataset', split=split)
# "

# echo "=== T5 Step 2: Pre-download flan-t5-small model & tokenizer ==="
# python3 -c "
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# AutoTokenizer.from_pretrained('google/flan-t5-small')
# AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
# "

# echo "=== T5 Step 3a: Pre-download ROUGE metric ==="
# python3 -c "import evaluate; evaluate.load('rouge')"

# echo "=== T5 Step 4: Pre-download NLTK punkt tokenizer ==="
# python3 -c "
# import nltk, os
# nltk_dir = os.path.join(os.getcwd(), '.nltk_data')
# os.makedirs(nltk_dir, exist_ok=True)
# nltk.download('punkt', download_dir=nltk_dir)
# nltk.download('punkt_tab', download_dir=nltk_dir)
# "

# ─── New: SARI metric (required for verify_onnx_export.py) ───────────────────

echo "=== Downloading SARI metric ==="
python3 -c "
import evaluate
print('Downloading sari metric...')
evaluate.load('sari')
print('sari metric cached.')
"

echo ""
echo "=== Done ==="
