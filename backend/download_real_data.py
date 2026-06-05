"""
Download and prepare ALL real-world datasets for fine-tuning BERT.

Combines FOUR real datasets with NO subsampling:
    1. Sp1786/multiclass-sentiment-analysis-dataset (HuggingFace) - ~241K
    2. stanfordnlp/sst2 (Stanford Sentiment Treebank) - ~67K
    3. NLPC-UOM/Student_feedback_analysis_dataset (HuggingFace)
    4. Our curated pedagogical training data (training_data.py) - ~2.1K

All data is used — no downsampling. Class imbalance is handled during
training with weighted loss.

Usage:
    python download_real_data.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "real_dataset.json")

LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_multiclass_sentiment() -> List[Dict[str, Any]]:
    """Sp1786/multiclass-sentiment-analysis-dataset.
    ~241K samples, 3-class (0=Neg, 1=Neu, 2=Pos)."""
    from datasets import load_dataset

    logger.info("[1/4] Loading Sp1786/multiclass-sentiment-analysis-dataset ...")
    try:
        ds = load_dataset(
            "Sp1786/multiclass-sentiment-analysis-dataset",
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.error("  FAILED: %s", exc)
        return []

    samples: List[Dict[str, Any]] = []
    for split_name in ds:
        split = ds[split_name]
        cols = split.column_names
        logger.info("  Split '%s': %d rows, columns: %s", split_name, len(split), cols)

        text_col = next((c for c in cols if c.lower() in ("text", "comment")), None)
        label_col = next((c for c in cols if c.lower() in ("label", "sentiment")), None)

        if not text_col or not label_col:
            logger.warning("  Could not find text/label columns in %s", cols)
            continue

        for row in split:
            text = str(row[text_col]).strip()
            label = int(row[label_col])
            if text and len(text) >= 5 and label in (0, 1, 2):
                samples.append({
                    "text": text,
                    "label": label,
                    "source": "multiclass_sentiment",
                })

    logger.info("  -> %d samples from multiclass-sentiment", len(samples))
    return samples


def load_stanford_sst() -> List[Dict[str, Any]]:
    """stanfordnlp/sst2 (Stanford Sentiment Treebank 2).
    ~67K samples, binary (0=Negative, 1=Positive).
    We map: 0->0 (Negative), 1->2 (Positive). No neutral from this source."""
    from datasets import load_dataset

    logger.info("[2/4] Loading stanfordnlp/sst2 (Stanford Sentiment Treebank) ...")
    try:
        ds = load_dataset("stanfordnlp/sst2", trust_remote_code=True)
    except Exception as exc:
        logger.error("  FAILED: %s", exc)
        return []

    samples: List[Dict[str, Any]] = []
    for split_name in ds:
        split = ds[split_name]
        cols = split.column_names
        logger.info("  Split '%s': %d rows, columns: %s", split_name, len(split), cols)

        text_col = next((c for c in cols if c.lower() in ("sentence", "text")), None)
        label_col = next((c for c in cols if c.lower() in ("label",)), None)

        if not text_col or not label_col:
            logger.warning("  Could not find text/label columns in %s", cols)
            continue

        for row in split:
            text = str(row[text_col]).strip()
            raw_label = row[label_col]

            if not text or len(text) < 5:
                continue

            # SST-2: 0=negative, 1=positive; map to our scheme
            if raw_label == -1:
                # Unlabelled samples in some splits
                continue
            elif raw_label == 0:
                label = 0  # Negative
            elif raw_label == 1:
                label = 2  # Positive
            else:
                continue

            samples.append({
                "text": text,
                "label": label,
                "source": "stanford_sst2",
            })

    logger.info("  -> %d samples from Stanford SST-2", len(samples))
    return samples


def load_student_feedback() -> List[Dict[str, Any]]:
    """NLPC-UOM/Student_feedback_analysis_dataset.
    Real student feedback with sentiment labels."""
    from datasets import load_dataset

    logger.info("[3/4] Loading NLPC-UOM/Student_feedback_analysis_dataset ...")
    try:
        ds = load_dataset(
            "NLPC-UOM/Student_feedback_analysis_dataset",
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.warning("  FAILED: %s (will proceed without)", exc)
        return []

    samples: List[Dict[str, Any]] = []
    for split_name in ds:
        split = ds[split_name]
        cols = split.column_names
        logger.info("  Split '%s': %d rows, columns: %s", split_name, len(split), cols)

        # Auto-detect columns
        text_col = None
        label_col = None
        for c in cols:
            cl = c.lower()
            if cl in ("text", "feedback", "comment", "sentence", "review"):
                text_col = c
            elif cl in ("label", "sentiment", "class", "category"):
                label_col = c

        if text_col is None:
            text_col = cols[0]
        if label_col is None and len(cols) > 1:
            label_col = cols[-1]

        if label_col is None:
            continue

        for row in split:
            text = str(row[text_col]).strip()
            raw_label = row[label_col]
            if not text or len(text) < 5:
                continue
            label = _map_label(raw_label)
            if label is not None:
                samples.append({
                    "text": text,
                    "label": label,
                    "source": "student_feedback",
                })

    logger.info("  -> %d samples from student feedback", len(samples))
    return samples


def _map_label(raw: Any) -> Optional[int]:
    """Map various label formats to 0/1/2."""
    if isinstance(raw, int):
        if raw in (0, 1, 2):
            return raw
        if raw == -1:
            return 0
        if raw in (1, 2):
            return 0
        if raw == 3:
            return 1
        if raw in (4, 5):
            return 2
        return None
    if isinstance(raw, float):
        return _map_label(int(raw))
    if isinstance(raw, str):
        low = raw.lower().strip()
        if low in ("positive", "pos", "good", "awesome", "excellent"):
            return 2
        elif low in ("negative", "neg", "bad", "poor", "awful", "terrible"):
            return 0
        elif low in ("neutral", "neu", "average", "mixed", "okay"):
            return 1
        try:
            return _map_label(int(low))
        except ValueError:
            pass
    return None


def load_pedagogical_data() -> List[Dict[str, Any]]:
    """Our curated pedagogical training data (2,160 samples)."""
    from training_data import get_dataset

    logger.info("[4/4] Loading curated pedagogical dataset ...")
    raw = get_dataset(seed=42)
    samples = [
        {"text": s["text"], "label": s["label"], "source": "pedagogical_curated"}
        for s in raw
    ]
    logger.info("  -> %d samples from pedagogical data", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Assembly – NO subsampling
# ---------------------------------------------------------------------------


def build_full_dataset() -> List[Dict[str, Any]]:
    """Build the combined dataset from ALL sources. NO downsampling."""
    import random
    random.seed(42)

    all_samples: List[Dict[str, Any]] = []

    # Load all four sources
    all_samples.extend(load_multiclass_sentiment())
    all_samples.extend(load_stanford_sst())
    all_samples.extend(load_student_feedback())
    all_samples.extend(load_pedagogical_data())

    # Deduplicate by normalised text
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for s in all_samples:
        key = s["text"].lower().strip()[:200]  # first 200 chars for dedup
        if key not in seen:
            seen.add(key)
            unique.append(s)

    logger.info("Before dedup: %d | After dedup: %d", len(all_samples), len(unique))

    random.shuffle(unique)
    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("  REAL DATASET BUILDER (FULL SIZE - NO SUBSAMPLING)")
    print("=" * 70)
    print()

    start = time.perf_counter()
    samples = build_full_dataset()
    elapsed = time.perf_counter() - start

    # Stats
    dist = Counter(s["label"] for s in samples)
    source_dist = Counter(s["source"] for s in samples)

    print(f"\n  TOTAL SAMPLES: {len(samples)}")
    print(f"\n  Label distribution:")
    for label_id in sorted(dist):
        pct = dist[label_id] / len(samples) * 100
        print(f"    {LABEL_NAMES[label_id]:>10}: {dist[label_id]:>8}  ({pct:.1f}%)")
    print(f"\n  Source distribution:")
    for src, cnt in source_dist.most_common():
        pct = cnt / len(samples) * 100
        print(f"    {src:>25}: {cnt:>8}  ({pct:.1f}%)")

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    file_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    logger.info("Saved to %s (%.1f MB)", OUTPUT_PATH, file_mb)

    print(f"\n  Download time : {elapsed:.1f}s")
    print(f"  File size     : {file_mb:.1f} MB")
    print(f"  Saved to      : {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
