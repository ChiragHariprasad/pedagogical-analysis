"""
Head-to-head comparison of the pretrained HuggingFace model vs. the
fine-tuned BERT model on the same held-out pedagogical sentiment test set.

Produces:
    * Per-model accuracy, F1, precision, recall (macro & per-class)
    * Confusion matrices
    * Side-by-side comparison table
    * Saves results to ``comparison_results.json``

Usage:
    python compare_models.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNED_DIR = os.path.join(SCRIPT_DIR, "fine_tuned_bert_sentiment")
META_PATH = os.path.join(FINETUNED_DIR, "test_meta.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "comparison_results.json")

LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
NUM_LABELS = 3


# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------


def load_test_data() -> Tuple[List[str], List[int]]:
    """Load the held-out test set saved by finetune_bert.py."""
    if not os.path.exists(META_PATH):
        logger.error("Test metadata not found at %s. Run finetune_bert.py first.", META_PATH)
        sys.exit(1)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return meta["test_texts"], meta["test_labels"]


# ---------------------------------------------------------------------------
# Pretrained model predictions
# ---------------------------------------------------------------------------


def _get_pretrained_model() -> Tuple[Any, str]:
    """Load the same pretrained model the pipeline uses (cascade fallback)."""
    from transformers import pipeline as hf_pipeline

    candidates = [
        "yangheng/deberta-v3-base-absa-v1.1",
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "distilbert-base-uncased-finetuned-sst-2-english",
    ]
    for model_name in candidates:
        try:
            logger.info("Attempting pretrained model: %s", model_name)
            pipe = hf_pipeline("sentiment-analysis", model=model_name, truncation=True)
            logger.info("Loaded: %s", model_name)
            return pipe, model_name
        except Exception as exc:
            logger.warning("Failed %s: %s", model_name, exc)
    raise RuntimeError("Could not load any pretrained model.")


def _normalise_pretrained_output(label: str, score: float, model_name: str) -> int:
    """Map pretrained model output to our 3-class label space (0/1/2).

    Mirrors the logic in ``nlp_pipeline.classify_sentiment()``.
    """
    if "nlptown" in model_name:
        # 5-star: "1 star" .. "5 stars"
        stars = int(label.split()[0])
        if stars >= 4:
            return 2  # Positive
        elif stars <= 2:
            return 0  # Negative
        else:
            return 1  # Neutral
    else:
        # Binary POSITIVE/NEGATIVE
        if label.upper() == "POSITIVE":
            return 2 if score >= 0.6 else 1
        else:
            return 0 if score >= 0.6 else 1


def predict_pretrained(texts: List[str]) -> Tuple[List[int], str, float]:
    """Run the pretrained model on *texts*, return (preds, model_name, elapsed)."""
    pipe, model_name = _get_pretrained_model()
    start = time.perf_counter()
    preds: List[int] = []
    for text in texts:
        result = pipe(text[:512])[0]
        pred = _normalise_pretrained_output(result["label"], result["score"], model_name)
        preds.append(pred)
    elapsed = time.perf_counter() - start
    return preds, model_name, elapsed


# ---------------------------------------------------------------------------
# Fine-tuned model predictions
# ---------------------------------------------------------------------------


def predict_finetuned(texts: List[str]) -> Tuple[List[int], float]:
    """Run the fine-tuned BERT model on *texts*, return (preds, elapsed)."""
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    logger.info("Loading fine-tuned model from %s ...", FINETUNED_DIR)
    tokenizer = BertTokenizer.from_pretrained(FINETUNED_DIR)
    model = BertForSequenceClassification.from_pretrained(FINETUNED_DIR)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    start = time.perf_counter()
    preds: List[int] = []

    # Batch inference for speed
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**enc)
        batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        preds.extend(batch_preds)

    elapsed = time.perf_counter() - start
    return preds, elapsed


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_model(
    name: str,
    preds: List[int],
    labels: List[int],
    elapsed: float,
) -> Dict[str, Any]:
    """Compute all metrics for one model."""
    target_names = [LABEL_NAMES[i] for i in range(NUM_LABELS)]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    prec = precision_score(labels, preds, average="macro")
    rec = recall_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_LABELS)))
    report = classification_report(
        labels, preds, target_names=target_names, output_dict=True
    )

    return {
        "model_name": name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "inference_time_seconds": round(elapsed, 2),
    }


def print_results(results: Dict[str, Any]) -> None:
    """Pretty-print metrics for one model."""
    print(f"\n  Model        : {results['model_name']}")
    print(f"  Accuracy     : {results['accuracy']:.4f}")
    print(f"  F1 (macro)   : {results['f1_macro']:.4f}")
    print(f"  Precision    : {results['precision_macro']:.4f}")
    print(f"  Recall       : {results['recall_macro']:.4f}")
    print(f"  Inference    : {results['inference_time_seconds']:.2f}s")
    print(f"\n  Confusion Matrix:")
    cm = np.array(results["confusion_matrix"])
    header = "            " + "  ".join(f"{LABEL_NAMES[i]:>8}" for i in range(NUM_LABELS))
    print(f"  {header}")
    for i in range(NUM_LABELS):
        row = "  ".join(f"{cm[i][j]:>8}" for j in range(NUM_LABELS))
        print(f"  {LABEL_NAMES[i]:>10}  {row}")

    print(f"\n  Per-class breakdown:")
    report = results["classification_report"]
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-' * 54}")
    for cls_name in [LABEL_NAMES[i] for i in range(NUM_LABELS)]:
        r = report[cls_name]
        print(
            f"  {cls_name:<12} {r['precision']:>10.4f} {r['recall']:>10.4f} "
            f"{r['f1-score']:>10.4f} {r['support']:>10.0f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print()
    print("#" * 70)
    print("  MODEL COMPARISON: Pretrained vs Fine-Tuned BERT")
    print("#" * 70)

    # Load test data
    test_texts, test_labels = load_test_data()
    logger.info("Loaded %d test samples.", len(test_texts))

    # ---- Pretrained model ----
    print("\n" + "=" * 70)
    print("  [A] PRETRAINED MODEL (HuggingFace)")
    print("=" * 70)
    pretrained_preds, pretrained_name, pretrained_time = predict_pretrained(test_texts)
    pretrained_results = evaluate_model(
        f"Pretrained: {pretrained_name}",
        pretrained_preds,
        test_labels,
        pretrained_time,
    )
    print_results(pretrained_results)

    # ---- Fine-tuned model ----
    print("\n" + "=" * 70)
    print("  [B] FINE-TUNED BERT (bert-base-uncased)")
    print("=" * 70)
    finetuned_preds, finetuned_time = predict_finetuned(test_texts)
    finetuned_results = evaluate_model(
        "Fine-Tuned: bert-base-uncased",
        finetuned_preds,
        test_labels,
        finetuned_time,
    )
    print_results(finetuned_results)

    # ---- Side-by-side comparison ----
    print("\n" + "#" * 70)
    print("  SIDE-BY-SIDE COMPARISON")
    print("#" * 70)
    print(f"\n  {'Metric':<20} {'Pretrained':>14} {'Fine-Tuned':>14} {'Winner':>12}")
    print(f"  {'-' * 62}")

    metrics = [
        ("Accuracy", "accuracy"),
        ("F1 (macro)", "f1_macro"),
        ("Precision", "precision_macro"),
        ("Recall", "recall_macro"),
    ]
    wins = {"pretrained": 0, "finetuned": 0}
    for display_name, key in metrics:
        p_val = pretrained_results[key]
        f_val = finetuned_results[key]
        if f_val > p_val:
            winner = "Fine-Tuned *"
            wins["finetuned"] += 1
        elif p_val > f_val:
            winner = "Pretrained *"
            wins["pretrained"] += 1
        else:
            winner = "Tie"
        print(f"  {display_name:<20} {p_val:>14.4f} {f_val:>14.4f} {winner:>12}")

    inf_winner = (
        "Fine-Tuned *"
        if finetuned_results["inference_time_seconds"]
        < pretrained_results["inference_time_seconds"]
        else "Pretrained *"
    )
    print(
        f"  {'Inference Time':<20} "
        f"{pretrained_results['inference_time_seconds']:>13.2f}s "
        f"{finetuned_results['inference_time_seconds']:>13.2f}s "
        f"{inf_winner:>12}"
    )

    print(f"\n  Overall: Pretrained wins {wins['pretrained']}, "
          f"Fine-Tuned wins {wins['finetuned']}")

    # ---- Sample disagreements ----
    print(f"\n  Sample predictions where models disagree:")
    print(f"  {'-' * 66}")
    disagreements = 0
    for i, (text, true_lbl, p_pred, f_pred) in enumerate(
        zip(test_texts, test_labels, pretrained_preds, finetuned_preds)
    ):
        if p_pred != f_pred and disagreements < 10:
            print(f"    Text: {text[:70]}...")
            print(
                f"    True: {LABEL_NAMES[true_lbl]}  |  "
                f"Pretrained: {LABEL_NAMES[p_pred]}  |  "
                f"Fine-Tuned: {LABEL_NAMES[f_pred]}"
            )
            correct = []
            if p_pred == true_lbl:
                correct.append("Pretrained")
            if f_pred == true_lbl:
                correct.append("Fine-Tuned")
            if correct:
                print(f"    -> Correct: {', '.join(correct)}")
            else:
                print(f"    -> Both wrong")
            print()
            disagreements += 1

    if disagreements == 0:
        print("    (No disagreements found -- models agree on all test samples!)")

    # ---- Save results ----
    comparison = {
        "pretrained": pretrained_results,
        "finetuned": finetuned_results,
        "test_set_size": len(test_labels),
        "summary": {
            "pretrained_wins": wins["pretrained"],
            "finetuned_wins": wins["finetuned"],
        },
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results saved to %s", RESULTS_PATH)

    print("#" * 70)
    print(f"  Results saved to: {RESULTS_PATH}")
    print("#" * 70)
    print()


if __name__ == "__main__":
    main()
