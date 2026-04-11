"""
Fine-tune T5-small for medical text simplification.

Trains a text-to-text model on the GEM/cochrane-simplification dataset
(supplemented with curated clinical pairs) to simplify complex medical
language into patient-friendly explanations.

Usage:
    python training/data/prepare_simplification.py   # prepare data first
    python training/simplification/finetune_t5.py

Output:
    training/simplification/output/  — fine-tuned model checkpoint
"""

import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate
import nltk

NLTK_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".nltk_data"
)
nltk.data.path.insert(0, NLTK_DATA_DIR)
for pkg in ("tokenizers/punkt", "tokenizers/punkt_tab"):
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg.split("/")[1], download_dir=NLTK_DATA_DIR, quiet=True)

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "simplification_pairs"
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MODEL_NAME = "google/flan-t5-small"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256
LEARNING_RATE = 3e-4
BATCH_SIZE = 16
NUM_EPOCHS = 30
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
EARLY_STOPPING_PATIENCE = 5


def preprocess(examples, tokenizer):
    """Tokenize input/target pairs with dynamic padding."""
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
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    return model_inputs


_rouge = None


def compute_metrics(eval_pred, tokenizer):
    """Compute ROUGE metrics for simplification quality."""
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


def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    for split in dataset:
        print(f"  {split}: {len(dataset[split])} examples")

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
        generation_max_length=MAX_TARGET_LENGTH,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        logging_steps=50,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        report_to="none",
        save_total_limit=3,
    )

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
        "Simplify this medical text for a patient: The patient presents with acute myocardial infarction with ST-segment elevation.",
        "Simplify this medical text for a patient: Labs reveal elevated creatinine at 2.8 mg/dL suggestive of acute kidney injury.",
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
