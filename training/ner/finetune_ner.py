"""
Fine-tune DistilBERT on the BC5CDR dataset for biomedical NER.

Trains a token classification model to identify Disease and Chemical entities
in medical text.

Usage:
    python training/data/prepare_bc5cdr.py   # prepare data first
    python training/ner/finetune_ner.py

Output:
    training/ner/output/  — fine-tuned model checkpoint
"""

import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import evaluate

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "bc5cdr_processed"
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MODEL_NAME = "distilbert-base-uncased"

LABEL_LIST = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

MAX_LENGTH = 512
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01


def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize inputs and align NER labels with sub-word tokens."""
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(labels[word_id])
            else:
                label_id = labels[word_id]
                if LABEL_LIST[label_id].startswith("B-"):
                    label_ids.append(label_id + 1)  # B- -> I-
                else:
                    label_ids.append(label_id)
            previous_word_id = word_id

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(eval_pred):
    """Compute precision, recall, F1 using seqeval."""
    seqeval = evaluate.load("seqeval")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    true_predictions = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_pred = []
        true_label = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_pred.append(LABEL_LIST[p])
                true_label.append(LABEL_LIST[l])
        true_predictions.append(true_pred)
        true_labels.append(true_label)

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    use_mps = torch.backends.mps.is_available()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=False,
        use_mps_device=use_mps,
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Test Results: {test_results}")

    best_model_path = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"\nBest model saved to {best_model_path}")


if __name__ == "__main__":
    main()
