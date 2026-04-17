#!/bin/bash
#SBATCH --job-name=cb-t5
#SBATCH --output=logs/preparedata_%j.out
#SBATCH --error=logs/preparedata_%j.err
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

export HF_HOME=/scratch/user/vinaysingh/.cache/huggingface
export PIP_CACHE_DIR=/scratch/user/vinaysingh/PIP_CACHE
export NLTK_DATA=/scratch/user/vinaysingh/Cognitive-Bridge/.nltk_data

cd /scratch/user/vinaysingh/Cognitive-Bridge

python training/data/prepare_simplification_medisimplifier.py
python training/data/verify_simplification_dataset.py --path training/data/simplification_pairs_medisimplifier