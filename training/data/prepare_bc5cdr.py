"""
Download and prepare the BC5CDR dataset for NER fine-tuning.

The BC5CDR corpus contains annotated biomedical text with Disease and Chemical
entities. This script loads the Parquet-native version from masaenger/bc5cdr
(bc5cdr_source config) and converts span-level annotations to BIO-tagged token
sequences for token classification training.

Usage:
    python training/data/prepare_bc5cdr.py

Output:
    training/data/bc5cdr_processed/ — HuggingFace Dataset on disk
"""

import os
import re
from datasets import Dataset, DatasetDict, load_dataset

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
    """Load the BC5CDR dataset using explicit Parquet file paths."""
    print("Loading BC5CDR dataset (masaenger/bc5cdr, parquet)...")
    base = "hf://datasets/masaenger/bc5cdr/bc5cdr_source"
    dataset = load_dataset("parquet", data_files={
        "train": f"{base}/train-00000-of-00001.parquet",
        "validation": f"{base}/validation-00000-of-00001.parquet",
        "test": f"{base}/test-00000-of-00001.parquet",
    })
    for split in dataset:
        print(f"  {split}: {len(dataset[split])} documents")
    return dataset


def tokenize_simple(text):
    """
    Split text into tokens and track character offsets for each token.
    Returns list of (token, start_char, end_char) tuples.
    """
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', text)]


def passage_to_bio(passage, passage_doc_offset=0):
    """
    Convert a single passage (with its entities) into tokens + BIO tags.

    bc5cdr_source stores entity offsets relative to the full document, not
    the individual passage.  ``passage_doc_offset`` is the character position
    where this passage begins in the document so we can adjust correctly.
    """
    text = passage["text"]
    if not text or not text.strip():
        return [], []

    token_spans = tokenize_simple(text)
    if not token_spans:
        return [], []

    tags = ["O"] * len(token_spans)

    entities = passage.get("entities", [])
    if not entities:
        tokens = [t[0] for t in token_spans]
        return tokens, tags

    intervals = []
    for ent in entities:
        ent_type = ent.get("type", "")
        if ent_type not in ("Chemical", "Disease"):
            continue
        ent_texts = ent.get("text", [])
        for idx, offset_pair in enumerate(ent.get("offsets", [])):
            if len(offset_pair) < 2:
                continue
            start = offset_pair[0] - passage_doc_offset
            end = offset_pair[1] - passage_doc_offset
            if start < 0 or end > len(text):
                continue
            expected = ent_texts[idx] if idx < len(ent_texts) else None
            if expected and text[start:end] != expected:
                fallback = text.find(expected)
                if fallback >= 0:
                    start, end = fallback, fallback + len(expected)
                else:
                    continue
            intervals.append((start, end, ent_type))

    intervals.sort(key=lambda x: x[0])

    for ent_start, ent_end, ent_type in intervals:
        first_token = True
        for i, (_, tok_start, tok_end) in enumerate(token_spans):
            if tok_end <= ent_start:
                continue
            if tok_start >= ent_end:
                break
            if tags[i] != "O":
                continue
            if first_token:
                tags[i] = f"B-{ent_type}"
                first_token = False
            else:
                tags[i] = f"I-{ent_type}"

    tokens = [t[0] for t in token_spans]
    return tokens, tags


def prepare_dataset():
    """Full preparation pipeline — one example per passage to avoid truncation."""
    raw_dataset = load_bc5cdr()

    converted_splits = {}
    for split_name in raw_dataset:
        print(f"\nConverting {split_name} split (passage-level)...")
        tokens_list = []
        tags_list = []

        for example in raw_dataset[split_name]:
            passages = example["passages"]
            doc_offset = 0
            for passage in passages:
                tokens, tags = passage_to_bio(passage, doc_offset)
                doc_offset += len(passage["text"]) + 1
                if tokens:
                    tag_ids = [LABEL2ID.get(tag, 0) for tag in tags]
                    tokens_list.append(tokens)
                    tags_list.append(tag_ids)

        converted_splits[split_name] = Dataset.from_dict({
            "tokens": tokens_list,
            "ner_tags": tags_list,
        })
        print(f"  {split_name}: {len(converted_splits[split_name])} examples")

        entity_count = sum(1 for tags in tags_list for t in tags if t != 0)
        print(f"  Entity tokens: {entity_count}")

    dataset = DatasetDict(converted_splits)

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
        print(f"  {token:30s} -> {LABEL_LIST[tag]}")


if __name__ == "__main__":
    if os.path.isfile(os.path.join(OUTPUT_DIR, "dataset_dict.json")):
        print(f"Processed dataset already exists at {OUTPUT_DIR}, skipping preparation.")
        print("Delete the directory to re-prepare: rm -rf " + OUTPUT_DIR)
    else:
        prepare_dataset()
