"""
Download and prepare the BC5CDR dataset for NER fine-tuning.

The BC5CDR corpus contains annotated biomedical text with Disease and Chemical
entities in BIO tagging format, suitable for token classification.

Usage:
    python training/data/prepare_bc5cdr.py

Output:
    training/data/bc5cdr_processed/ — HuggingFace Dataset on disk
"""

import os
from datasets import load_dataset, DatasetDict

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bc5cdr_processed")

LABEL_LIST = [
    "O",
    "B-Chemical",
    "I-Chemical",
    "B-Disease",
    "I-Disease",
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def load_bc5cdr():
    """Load the BC5CDR dataset from HuggingFace datasets hub."""
    print("Loading BC5CDR dataset...")
    dataset = load_dataset("tner/bc5cdr")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")
    return dataset


def inspect_labels(dataset):
    """Print unique labels in the dataset."""
    all_labels = set()
    for split in dataset:
        for example in dataset[split]:
            all_labels.update(example["tags"])
    print(f"Unique label IDs found: {sorted(all_labels)}")


def remap_labels(example):
    """Ensure labels are mapped to our standard label set."""
    tag_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    example["ner_tags"] = [tag_mapping.get(t, 0) for t in example["tags"]]
    return example


def prepare_dataset():
    """Full preparation pipeline."""
    dataset = load_bc5cdr()
    inspect_labels(dataset)

    dataset = dataset.map(remap_labels)

    dataset = dataset.remove_columns(
        [col for col in dataset["train"].column_names if col not in ["tokens", "ner_tags"]]
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset.save_to_disk(OUTPUT_DIR)
    print(f"\nProcessed dataset saved to {OUTPUT_DIR}")

    with open(os.path.join(OUTPUT_DIR, "label_info.txt"), "w") as f:
        f.write("Label List:\n")
        for label_id, label_name in ID2LABEL.items():
            f.write(f"  {label_id}: {label_name}\n")

    print("\nSample from training set:")
    sample = dataset["train"][0]
    tokens = sample["tokens"]
    tags = sample["ner_tags"]
    for token, tag in zip(tokens[:20], tags[:20]):
        print(f"  {token:20s} -> {LABEL_LIST[tag]}")


if __name__ == "__main__":
    prepare_dataset()
