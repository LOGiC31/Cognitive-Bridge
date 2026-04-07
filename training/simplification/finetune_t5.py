"""
Fine-tune T5-small for medical text simplification.

Trains a text-to-text model to simplify complex medical language into
patient-friendly explanations.

Usage:
    python training/data/prepare_simplification.py   # prepare data first
    python training/simplification/finetune_t5.py

Output:
    training/simplification/output/  — fine-tuned model checkpoint
"""

import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import evaluate
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "simplification_pairs"
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MODEL_NAME = "t5-small"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
LEARNING_RATE = 3e-4
BATCH_SIZE = 8
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01


def preprocess(examples, tokenizer):
    """Tokenize input/target pairs."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    return model_inputs


def compute_metrics(eval_pred, tokenizer):
    """Compute ROUGE metrics for simplification quality."""
    rouge = evaluate.load("rouge")
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    return {k: round(v * 100, 4) for k, v in result.items()}


def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)

    print(f"Loading tokenizer: {MODEL_NAME}")
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

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        logging_steps=10,
        fp16=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating on validation set...")
    val_results = trainer.evaluate()
    print(f"Validation Results: {val_results}")

    best_model_path = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"\nBest model saved to {best_model_path}")


if __name__ == "__main__":
    main()
