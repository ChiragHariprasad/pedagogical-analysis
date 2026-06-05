"""
Fine-tune ``bert-base-uncased`` on the FULL real-world dataset with
energy consumption monitoring and comprehensive training report.

Features:
    * Trains on the entire real_dataset.json (300K+ samples)
    * Monitors CPU usage, memory, and estimates energy consumption
    * Generates a detailed Markdown training report
    * Tracks per-epoch metrics, loss curves
    * Saves model + tokenizer + report on completion

Usage:
    python finetune_bert_real.py
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
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
    TrainerCallback,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "bert-base-uncased"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "real_dataset.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "fine_tuned_bert_real")
REPORT_PATH = os.path.join(SCRIPT_DIR, "training_report.md")
RESULTS_JSON_PATH = os.path.join(SCRIPT_DIR, "training_results.json")

NUM_LABELS = 3
MAX_LENGTH = 128
TEST_SIZE = 0.10   # 10% test — keeps training set as large as possible
RANDOM_SEED = 42

# Training hyper-params (CPU-optimized)
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRADIENT_ACCUMULATION = 4  # effective batch = 32
LOGGING_STEPS = 500

LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Energy & Resource Monitor
# ---------------------------------------------------------------------------


class EnergyMonitor:
    """Background thread that samples CPU/memory stats every N seconds
    and estimates energy consumption."""

    def __init__(self, sample_interval: float = 5.0, cpu_tdp_watts: float = 65.0):
        self.sample_interval = sample_interval
        self.cpu_tdp_watts = cpu_tdp_watts  # Typical desktop CPU TDP
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Recorded data
        self.cpu_samples: List[float] = []       # CPU percent (0-100)
        self.memory_samples: List[float] = []    # RAM usage in MB
        self.timestamps: List[float] = []        # epoch time
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def start(self) -> None:
        self.start_time = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info("Energy monitor started (sampling every %.0fs)", self.sample_interval)

    def stop(self) -> Dict[str, Any]:
        self._running = False
        self.end_time = time.time()
        if self._thread:
            self._thread.join(timeout=10)
        return self.get_report()

    def _sample_loop(self) -> None:
        while self._running:
            try:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                self.cpu_samples.append(cpu)
                self.memory_samples.append(mem)
                self.timestamps.append(time.time())
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def get_report(self) -> Dict[str, Any]:
        duration_s = self.end_time - self.start_time
        duration_h = duration_s / 3600.0

        if not self.cpu_samples:
            return {"error": "No samples collected"}

        avg_cpu = np.mean(self.cpu_samples)
        max_cpu = np.max(self.cpu_samples)
        min_cpu = np.min(self.cpu_samples)

        avg_mem = np.mean(self.memory_samples)
        max_mem = np.max(self.memory_samples)

        # Energy estimation:
        # Power = TDP * (avg_cpu_usage / 100) * scaling_factor
        # Scaling factor accounts for non-linear CPU power curves
        # At 100% CPU, power ~ TDP; at 50%, power ~ 60% TDP (not linear)
        scaling = 0.3 + 0.7 * (avg_cpu / 100.0)  # idle=30% TDP, full=100% TDP
        avg_power_w = self.cpu_tdp_watts * scaling

        # Add estimated RAM power (~3W per 8GB stick, scale by usage)
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        ram_power_w = (total_ram_gb / 8.0) * 3.0

        total_power_w = avg_power_w + ram_power_w
        energy_wh = total_power_w * duration_h
        energy_kwh = energy_wh / 1000.0

        # CO2 estimation (global average: ~475g CO2/kWh)
        co2_grams = energy_kwh * 475.0
        # India average: ~720g CO2/kWh
        co2_grams_india = energy_kwh * 720.0

        return {
            "duration_seconds": round(duration_s, 1),
            "duration_formatted": str(timedelta(seconds=int(duration_s))),
            "num_samples_collected": len(self.cpu_samples),
            "cpu": {
                "average_percent": round(avg_cpu, 1),
                "max_percent": round(max_cpu, 1),
                "min_percent": round(min_cpu, 1),
                "num_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
            },
            "memory": {
                "average_mb": round(avg_mem, 1),
                "peak_mb": round(max_mem, 1),
                "total_system_gb": round(total_ram_gb, 1),
            },
            "energy": {
                "estimated_cpu_tdp_watts": self.cpu_tdp_watts,
                "estimated_avg_power_watts": round(total_power_w, 1),
                "estimated_cpu_power_watts": round(avg_power_w, 1),
                "estimated_ram_power_watts": round(ram_power_w, 1),
                "total_energy_wh": round(energy_wh, 2),
                "total_energy_kwh": round(energy_kwh, 4),
            },
            "carbon": {
                "co2_grams_global_avg": round(co2_grams, 2),
                "co2_grams_india_avg": round(co2_grams_india, 2),
            },
        }


# ---------------------------------------------------------------------------
# Training metrics callback
# ---------------------------------------------------------------------------


class MetricsLogger(TrainerCallback):
    """Logs per-step and per-epoch metrics for the training report."""

    def __init__(self):
        self.step_losses: List[Dict[str, Any]] = []
        self.epoch_metrics: List[Dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 2)}
            entry.update({k: v for k, v in logs.items()
                          if isinstance(v, (int, float))})
            self.step_losses.append(entry)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 2)}
            entry.update({k: round(v, 4) if isinstance(v, float) else v
                          for k, v in metrics.items()})
            self.epoch_metrics.append(entry)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class SentimentDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
    }


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def generate_report(
    dataset_stats: Dict[str, Any],
    train_result: Any,
    eval_metrics: Dict[str, Any],
    test_labels: List[int],
    test_preds: np.ndarray,
    energy_report: Dict[str, Any],
    metrics_logger: MetricsLogger,
    training_time_s: float,
    system_info: Dict[str, Any],
) -> str:
    """Generate a comprehensive Markdown training report."""

    target_names = [LABEL_NAMES[i] for i in range(NUM_LABELS)]
    cls_report = classification_report(
        test_labels, test_preds, target_names=target_names
    )
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(NUM_LABELS)))
    acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average="macro")
    prec = precision_score(test_labels, test_preds, average="macro")
    rec = recall_score(test_labels, test_preds, average="macro")

    # Build epoch table
    epoch_rows = ""
    for em in metrics_logger.epoch_metrics:
        epoch_rows += (
            f"| {em.get('epoch', '?')} "
            f"| {em.get('eval_loss', '?'):.4f} "
            f"| {em.get('eval_accuracy', '?'):.4f} "
            f"| {em.get('eval_f1_macro', '?'):.4f} "
            f"| {em.get('eval_precision_macro', '?'):.4f} "
            f"| {em.get('eval_recall_macro', '?'):.4f} |\n"
        ) if isinstance(em.get('eval_loss'), float) else ""

    # Loss progression from step logs
    loss_entries = [s for s in metrics_logger.step_losses if "loss" in s]
    loss_table = ""
    # Show ~20 evenly spaced entries
    if loss_entries:
        step_size = max(1, len(loss_entries) // 20)
        for i in range(0, len(loss_entries), step_size):
            e = loss_entries[i]
            lr = e.get('learning_rate', '?')
            lr_str = f"{lr:.2e}" if isinstance(lr, float) else str(lr)
            loss_table += f"| {e['step']} | {e.get('epoch', '?')} | {e['loss']:.4f} | {lr_str} |\n"

    # Energy section
    en = energy_report.get("energy", {})
    cpu_info = energy_report.get("cpu", {})
    mem_info = energy_report.get("memory", {})
    carbon = energy_report.get("carbon", {})

    report = f"""# BERT Fine-Tuning Training Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model**: `{MODEL_NAME}` -> 3-class sentiment (Negative / Neutral / Positive)

---

## 1. System Information

| Parameter | Value |
|-----------|-------|
| OS | {system_info.get('os', 'N/A')} |
| CPU | {system_info.get('cpu', 'N/A')} |
| CPU Cores (physical/logical) | {cpu_info.get('physical_cores', '?')}/{cpu_info.get('num_cores', '?')} |
| Total RAM | {mem_info.get('total_system_gb', '?')} GB |
| Python | {system_info.get('python', 'N/A')} |
| PyTorch | {system_info.get('pytorch', 'N/A')} |
| Device | {system_info.get('device', 'CPU')} |
| GPU | {'None (CPU-only training)' if system_info.get('device') == 'cpu' else system_info.get('gpu', 'N/A')} |

---

## 2. Dataset Statistics

| Parameter | Value |
|-----------|-------|
| Total samples | {dataset_stats['total']:,} |
| Training samples | {dataset_stats['train_size']:,} |
| Test samples | {dataset_stats['test_size']:,} |
| Train/Test split | {100-TEST_SIZE*100:.0f}% / {TEST_SIZE*100:.0f}% |

### Label Distribution (Full Dataset)

| Label | Count | Percentage |
|-------|-------|------------|
| Negative | {dataset_stats['label_dist'].get(0, 0):,} | {dataset_stats['label_pct'].get(0, 0):.1f}% |
| Neutral | {dataset_stats['label_dist'].get(1, 0):,} | {dataset_stats['label_pct'].get(1, 0):.1f}% |
| Positive | {dataset_stats['label_dist'].get(2, 0):,} | {dataset_stats['label_pct'].get(2, 0):.1f}% |

### Data Sources

| Source | Count | Percentage |
|--------|-------|------------|
"""

    for src, cnt in sorted(dataset_stats['source_dist'].items(), key=lambda x: -x[1]):
        pct = cnt / dataset_stats['total'] * 100
        report += f"| {src} | {cnt:,} | {pct:.1f}% |\n"

    report += f"""
---

## 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `{MODEL_NAME}` |
| Max Sequence Length | {MAX_LENGTH} |
| Epochs | {EPOCHS} |
| Batch Size (per device) | {BATCH_SIZE} |
| Gradient Accumulation Steps | {GRADIENT_ACCUMULATION} |
| Effective Batch Size | {BATCH_SIZE * GRADIENT_ACCUMULATION} |
| Learning Rate | {LEARNING_RATE} |
| Weight Decay | {WEIGHT_DECAY} |
| Warmup Ratio | {WARMUP_RATIO} |
| Optimizer | AdamW |
| LR Scheduler | Linear with warmup |

---

## 4. Training Results

### Final Test Set Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **{acc:.4f}** ({acc*100:.2f}%) |
| **F1 (macro)** | **{f1:.4f}** |
| **Precision (macro)** | **{prec:.4f}** |
| **Recall (macro)** | **{rec:.4f}** |
| Training Loss (final) | {train_result.training_loss:.4f} |

### Per-Class Performance

```
{cls_report}
```

### Confusion Matrix

```
              Predicted
              Negative    Neutral    Positive
Negative    {cm[0][0]:>8}   {cm[0][1]:>8}   {cm[0][2]:>8}
Neutral     {cm[1][0]:>8}   {cm[1][1]:>8}   {cm[1][2]:>8}
Positive    {cm[2][0]:>8}   {cm[2][1]:>8}   {cm[2][2]:>8}
```

### Per-Epoch Evaluation

| Epoch | Loss | Accuracy | F1 (macro) | Precision | Recall |
|-------|------|----------|------------|-----------|--------|
{epoch_rows}

### Training Loss Progression

| Step | Epoch | Loss | Learning Rate |
|------|-------|------|---------------|
{loss_table}

---

## 5. Training Time

| Metric | Value |
|--------|-------|
| **Total Training Time** | **{energy_report.get('duration_formatted', 'N/A')}** |
| Total Seconds | {energy_report.get('duration_seconds', 'N/A')} |
| Samples/Second | {dataset_stats['train_size'] * EPOCHS / training_time_s:.1f} |
| Steps/Second | {train_result.metrics.get('train_steps_per_second', 'N/A')} |

---

## 6. Energy Consumption & Carbon Footprint

### Resource Utilization

| Metric | Value |
|--------|-------|
| Avg CPU Usage | {cpu_info.get('average_percent', '?')}% |
| Peak CPU Usage | {cpu_info.get('max_percent', '?')}% |
| Avg Process Memory | {mem_info.get('average_mb', '?'):.0f} MB |
| Peak Process Memory | {mem_info.get('peak_mb', '?'):.0f} MB |
| Monitoring Samples | {energy_report.get('num_samples_collected', '?')} |

### Energy Estimates

| Metric | Value |
|--------|-------|
| Estimated CPU TDP | {en.get('estimated_cpu_tdp_watts', '?')} W |
| Estimated Avg CPU Power | {en.get('estimated_cpu_power_watts', '?')} W |
| Estimated RAM Power | {en.get('estimated_ram_power_watts', '?')} W |
| **Estimated Total Avg Power** | **{en.get('estimated_avg_power_watts', '?')} W** |
| **Total Energy Consumed** | **{en.get('total_energy_wh', '?')} Wh ({en.get('total_energy_kwh', '?')} kWh)** |

### Carbon Footprint

| Region | CO2 Emissions |
|--------|--------------|
| Global Average (475g CO2/kWh) | {carbon.get('co2_grams_global_avg', '?')} g CO2 |
| India Average (720g CO2/kWh) | {carbon.get('co2_grams_india_avg', '?')} g CO2 |

> Note: Energy estimates are approximate, based on CPU utilization and TDP.
> Actual consumption may vary based on hardware power management, ambient
> temperature, and other system workloads.

---

## 7. Model Artifacts

| File | Description |
|------|-------------|
| `fine_tuned_bert_real/` | Saved model weights & tokenizer |
| `fine_tuned_bert_real/test_meta.json` | Test set data for comparison |
| `training_report.md` | This report |
| `training_results.json` | Raw metrics in JSON format |
| `real_dataset.json` | Full training dataset |

---

*Report generated by the Pedagogical Intelligence System BERT fine-tuning pipeline.*
"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    overall_start = time.time()

    print("=" * 70)
    print("  BERT FINE-TUNING ON FULL REAL-WORLD DATASET")
    print("  with Energy Monitoring & Training Report")
    print("=" * 70)

    # ---- System info ----
    system_info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor() or "Unknown",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        system_info["gpu"] = torch.cuda.get_device_name(0)

    device = system_info["device"]
    logger.info("Device: %s", device)

    # ---- 1. Load data ----
    logger.info("Loading dataset from %s ...", DATA_PATH)
    if not os.path.exists(DATA_PATH):
        logger.error("Dataset not found! Run: python download_real_data.py")
        sys.exit(1)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    texts = [s["text"] for s in raw_data]
    labels = [s["label"] for s in raw_data]
    sources = [s.get("source", "unknown") for s in raw_data]

    from collections import Counter
    label_dist = dict(Counter(labels))
    source_dist = dict(Counter(sources))
    label_pct = {k: v / len(labels) * 100 for k, v in label_dist.items()}

    dataset_stats = {
        "total": len(texts),
        "label_dist": label_dist,
        "label_pct": label_pct,
        "source_dist": source_dist,
    }

    logger.info("Total: %d samples", len(texts))
    for lbl in sorted(label_dist):
        logger.info("  %s: %d (%.1f%%)", LABEL_NAMES[lbl], label_dist[lbl], label_pct[lbl])

    # ---- 2. Train/test split ----
    (
        train_texts, test_texts,
        train_labels, test_labels,
        train_idx, test_idx,
    ) = train_test_split(
        texts, labels, range(len(texts)),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    dataset_stats["train_size"] = len(train_texts)
    dataset_stats["test_size"] = len(test_texts)
    logger.info("Train: %d | Test: %d", len(train_texts), len(test_texts))

    # ---- 3. Tokenize ----
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    logger.info("Tokenizing %d training samples ...", len(train_texts))
    train_enc = tokenizer(
        train_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )
    logger.info("Tokenizing %d test samples ...", len(test_texts))
    test_enc = tokenizer(
        test_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )

    train_dataset = SentimentDataset(train_enc, train_labels)
    test_dataset = SentimentDataset(test_enc, test_labels)

    # ---- 4. Load model ----
    logger.info("Loading model: %s (num_labels=%d)", MODEL_NAME, NUM_LABELS)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    # ---- 5. Compute class weights for imbalanced data ----
    total = len(train_labels)
    class_counts = Counter(train_labels)
    # Inverse frequency weighting
    weights = torch.tensor([
        total / (NUM_LABELS * class_counts.get(i, 1))
        for i in range(NUM_LABELS)
    ], dtype=torch.float32)
    logger.info("Class weights: %s", weights.tolist())

    # ---- 6. Training args ----
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
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=False,
        seed=RANDOM_SEED,
        disable_tqdm=False,
    )

    # ---- 7. Custom Trainer with weighted loss ----
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_t = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(logits.device))
            loss = loss_fn(logits, labels_t)
            return (loss, outputs) if return_outputs else loss

    metrics_logger = MetricsLogger()

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[metrics_logger],
    )

    # ---- 8. Start energy monitor & train ----
    energy_monitor = EnergyMonitor(sample_interval=5.0, cpu_tdp_watts=65.0)
    energy_monitor.start()

    logger.info("=" * 60)
    logger.info("  STARTING TRAINING on %d samples, %d epochs", len(train_texts), EPOCHS)
    logger.info("  Estimated time on CPU: %.1f hours",
                (len(train_texts) / (BATCH_SIZE * GRADIENT_ACCUMULATION)) * EPOCHS / 3600)
    logger.info("=" * 60)

    train_start = time.time()
    train_result = trainer.train()
    training_time_s = time.time() - train_start

    logger.info("Training completed in %s", str(timedelta(seconds=int(training_time_s))))

    # ---- 9. Stop energy monitor ----
    energy_report = energy_monitor.stop()

    # ---- 10. Final evaluation ----
    logger.info("Running final evaluation on test set ...")
    eval_metrics = trainer.evaluate()
    logger.info("Eval: %s", eval_metrics)

    preds_output = trainer.predict(test_dataset)
    test_preds = np.argmax(preds_output.predictions, axis=-1)

    # ---- 11. Save model ----
    logger.info("Saving model to %s ...", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save test metadata for comparison script
    test_meta = {
        "test_indices": list(test_idx),
        "test_labels": test_labels,
        "test_texts": test_texts,
        "label_names": LABEL_NAMES,
        "eval_metrics": {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in eval_metrics.items()
        },
        "training_time_seconds": round(training_time_s, 1),
        "dataset_size": len(texts),
    }
    meta_path = os.path.join(OUTPUT_DIR, "test_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(test_meta, f, indent=2, ensure_ascii=False)

    # ---- 12. Generate report ----
    logger.info("Generating training report ...")
    report_md = generate_report(
        dataset_stats=dataset_stats,
        train_result=train_result,
        eval_metrics=eval_metrics,
        test_labels=test_labels,
        test_preds=test_preds,
        energy_report=energy_report,
        metrics_logger=metrics_logger,
        training_time_s=training_time_s,
        system_info=system_info,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Report saved to %s", REPORT_PATH)

    # Save raw results JSON
    results_json = {
        "system_info": system_info,
        "dataset_stats": {k: v for k, v in dataset_stats.items()
                          if k != "source_dist" or True},
        "training_config": {
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
        },
        "final_metrics": {
            "accuracy": round(float(accuracy_score(test_labels, test_preds)), 4),
            "f1_macro": round(float(f1_score(test_labels, test_preds, average="macro")), 4),
            "precision_macro": round(float(precision_score(test_labels, test_preds, average="macro")), 4),
            "recall_macro": round(float(recall_score(test_labels, test_preds, average="macro")), 4),
            "training_loss": round(train_result.training_loss, 4),
        },
        "training_time": {
            "total_seconds": round(training_time_s, 1),
            "formatted": str(timedelta(seconds=int(training_time_s))),
        },
        "energy_report": energy_report,
        "epoch_metrics": metrics_logger.epoch_metrics,
        "confusion_matrix": cm.tolist() if hasattr(cm, 'tolist') else confusion_matrix(test_labels, test_preds, labels=list(range(NUM_LABELS))).tolist(),
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results JSON saved to %s", RESULTS_JSON_PATH)

    # ---- 13. Print summary ----
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Dataset      : {len(texts):,} samples")
    print(f"  Training time: {energy_report.get('duration_formatted', 'N/A')}")
    print(f"  Accuracy     : {accuracy_score(test_labels, test_preds):.4f}")
    print(f"  F1 (macro)   : {f1_score(test_labels, test_preds, average='macro'):.4f}")
    print(f"  Energy       : {en.get('total_energy_wh', '?')} Wh ({en.get('total_energy_kwh', '?')} kWh)")
    print(f"  CO2 (India)  : {carbon.get('co2_grams_india_avg', '?')} g")
    print(f"  Report       : {REPORT_PATH}")
    print(f"  Model        : {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
