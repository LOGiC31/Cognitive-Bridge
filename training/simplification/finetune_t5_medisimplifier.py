"""
Fine-tune T5-small for medical text simplification (Medisimplifier dataset).

This script is identical in spirit to `finetune_t5.py` but points to:
  training/data/simplification_pairs_medisimplifier

Usage:
  python training/data/prepare_simplification_medisimplifier.py
  python training/simplification/finetune_t5_medisimplifier.py

Output:
  training/simplification/output_medisimplifier/ — fine-tuned model checkpoint
"""

import os
import random

import evaluate
import nltk
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

NLTK_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".nltk_data",
)
nltk.data.path.insert(0, NLTK_DATA_DIR)
for pkg in ("tokenizers/punkt", "tokenizers/punkt_tab"):
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg.split("/")[1], download_dir=NLTK_DATA_DIR, quiet=True)

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "simplification_pairs_medisimplifier",
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_medisimplifier")
MODEL_NAME = "google/flan-t5-small"

# Dataset is now section-level pairs (avg ~115 input tokens, ~100 target tokens).
# p99 input tokens = ~430, p99 target tokens = ~395 — both fit within T5-small's
# 512-token architectural limit with no truncation needed.
# verify_simplification_dataset.py recommends MAX_TARGET_LENGTH ~410 to cover p99.
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512

LEARNING_RATE = 3e-4
BATCH_SIZE = 16
NUM_EPOCHS = 30
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
EARLY_STOPPING_PATIENCE = 5


def preprocess(examples, tokenizer):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    return model_inputs


_rouge = None


def compute_metrics(eval_pred, tokenizer):
    global _rouge
    if _rouge is None:
        _rouge = evaluate.load("rouge")

    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = _rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    return {k: round(v * 100, 4) for k, v in result.items()}


def print_dataset_sanity(dataset):
    print("\nDataset sanity-check (random samples):")
    rng = random.Random(7)
    for split in ("train", "validation", "test"):
        if split not in dataset:
            continue
        d = dataset[split]
        print(f"- {split}: {len(d)} examples")
        if len(d) == 0:
            continue
        for idx in rng.sample(range(len(d)), k=min(2, len(d))):
            ex = d[int(idx)]
            print("  INPUT :", ex["input_text"][:240].replace("\n", " ") + ("…" if len(ex["input_text"]) > 240 else ""))
            print("  TARGET:", ex["target_text"][:240].replace("\n", " ") + ("…" if len(ex["target_text"]) > 240 else ""))
            print("  ---")


def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    for split in dataset:
        print(f"  {split}: {len(dataset[split])} examples")
    print_dataset_sanity(dataset)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    use_cuda = torch.cuda.is_available()

    # Use available CPUs for DataLoader workers (SLURM allocates 8 CPUs; keep 1 for main process).
    num_workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1)

    use_bf16 = use_cuda and torch.cuda.get_device_capability()[0] >= 8  # A100 supports bf16

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        predict_with_generate=True,
        dataloader_pin_memory=use_cuda,
        dataloader_num_workers=num_workers,
        generation_max_length=MAX_TARGET_LENGTH,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        logging_steps=50,
        bf16=use_bf16,
        report_to="none",
        save_total_limit=3,
    )
    print(f"\nTraining config: bf16={use_bf16}, dataloader_num_workers={num_workers}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    print(f"\nStarting training (early stopping patience={EARLY_STOPPING_PATIENCE})...")
    trainer.train()

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized["test"])
    print(f"Test Results: {test_results}")

    best_model_path = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"\nBest model saved to {best_model_path}")

    print("\nSample predictions:")
    sample_inputs = [
        "Simplify this medical text for a patient: Patient presents with persistent dyspnea on exertion and bilateral lower extremity edema.",
        "Simplify this medical text for a patient: Hemoglobin A1c of 9.2% indicates poorly controlled diabetes mellitus type 2 despite metformin.",
    ]
    inputs = tokenizer(sample_inputs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH)
    if use_cuda:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_length=MAX_TARGET_LENGTH)
    for inp, out in zip(sample_inputs, tokenizer.batch_decode(outputs, skip_special_tokens=True)):
        print(f"  Input:  {inp}")
        print(f"  Output: {out}")
        print()


if __name__ == "__main__":
    main()

