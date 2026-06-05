"""
Fine-tune ``bert-base-uncased`` for 3-class pedagogical sentiment analysis.

Classes:  0 = Negative, 1 = Neutral, 2 = Positive

Steps:
    1. Load labelled data from ``training_data.get_dataset()``
    2. Stratified 80/20 train / test split
    3. Tokenize with ``BertTokenizer``
    4. Train with HuggingFace ``Trainer`` (AdamW, linear LR schedule)
    5. Evaluate on held-out test set
    6. Save fine-tuned model + tokenizer + test indices to disk

Usage:
    python finetune_bert.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from training_data import LABEL_NAMES, get_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fine_tuned_bert_sentiment")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
NUM_LABELS = 3
MAX_LENGTH = 128
TEST_SIZE = 0.20
RANDOM_SEED = 42

# Training hyper-parameters (tuned for CPU)
EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION = 2  # effective batch = 16

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------


class SentimentDataset(TorchDataset):
    """Wraps tokenized encodings + labels for the HuggingFace Trainer."""

    def __init__(self, encodings: Dict[str, Any], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Metric computation callback
# ---------------------------------------------------------------------------


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """Compute accuracy, macro-F1, precision, recall for the Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.perf_counter()

    # ---- 1. Load data ----
    logger.info("Loading dataset ...")
    raw = get_dataset(seed=RANDOM_SEED)
    texts = [s["text"] for s in raw]
    labels = [s["label"] for s in raw]
    logger.info("  Total samples: %d", len(texts))

    # ---- 2. Stratified split ----
    (
        train_texts,
        test_texts,
        train_labels,
        test_labels,
        train_idx,
        test_idx,
    ) = train_test_split(
        texts,
        labels,
        range(len(texts)),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    logger.info("  Train: %d  |  Test: %d", len(train_texts), len(test_texts))

    # ---- 3. Tokenize ----
    logger.info("Loading tokenizer: %s ...", MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    logger.info("Tokenizing ...")
    train_enc = tokenizer(
        train_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )
    test_enc = tokenizer(
        test_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )

    train_dataset = SentimentDataset(train_enc, train_labels)
    test_dataset = SentimentDataset(test_enc, test_labels)

    # ---- 4. Load model ----
    logger.info("Loading model: %s (num_labels=%d) ...", MODEL_NAME, NUM_LABELS)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ---- 5. Training arguments ----
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir=LOG_DIR,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",  # no wandb etc.
        fp16=False,  # CPU-safe
        seed=RANDOM_SEED,
        disable_tqdm=False,
    )

    # ---- 6. Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ---- 7. Train ----
    logger.info("=" * 60)
    logger.info("  STARTING TRAINING")
    logger.info("=" * 60)
    train_result = trainer.train()
    train_time = time.perf_counter() - start

    logger.info("Training completed in %.1f seconds.", train_time)
    logger.info("  Train loss : %.4f", train_result.training_loss)

    # ---- 8. Evaluate ----
    logger.info("Evaluating on test set ...")
    eval_metrics = trainer.evaluate()
    logger.info("  Eval results: %s", eval_metrics)

    # ---- 9. Detailed classification report ----
    preds_output = trainer.predict(test_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)

    print("\n" + "=" * 60)
    print("  FINE-TUNED BERT – TEST SET RESULTS")
    print("=" * 60)
    print(f"\n  Accuracy : {accuracy_score(test_labels, preds):.4f}")
    print(f"  F1 Macro : {f1_score(test_labels, preds, average='macro'):.4f}")
    print(f"  Precision: {precision_score(test_labels, preds, average='macro'):.4f}")
    print(f"  Recall   : {recall_score(test_labels, preds, average='macro'):.4f}")
    print(f"\n  Classification Report:\n")
    target_names = [LABEL_NAMES[i] for i in range(NUM_LABELS)]
    print(classification_report(test_labels, preds, target_names=target_names))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(test_labels, preds)
    print(f"  {cm}\n")

    # ---- 10. Save model + tokenizer ----
    logger.info("Saving model to %s ...", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save test indices and labels for the comparison script
    test_meta = {
        "test_indices": list(test_idx),
        "test_labels": test_labels,
        "test_texts": test_texts,
        "label_names": LABEL_NAMES,
        "eval_metrics": {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in eval_metrics.items()
        },
        "training_time_seconds": round(train_time, 1),
    }
    meta_path = os.path.join(OUTPUT_DIR, "test_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(test_meta, f, indent=2, ensure_ascii=False)
    logger.info("Test metadata saved to %s", meta_path)

    print("=" * 60)
    print(f"  Model saved to: {OUTPUT_DIR}")
    print(f"  Training time : {train_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
