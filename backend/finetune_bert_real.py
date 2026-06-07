"""
Fine-tune ``bert-base-uncased`` on the FULL real-world dataset with
energy consumption monitoring and comprehensive training report.

Features:
    * Trains on the entire real_dataset.json
    * Uses CPU-friendly fast tokenization and dynamic per-batch padding
    * Monitors CPU, process CPU time, memory, temperature, and energy estimates
    * Generates a detailed Markdown training report with SVG graphs
    * Tracks per-epoch metrics, loss curves
    * Saves model + tokenizer + report on completion

Usage:
    python finetune_bert_real.py
"""

from __future__ import annotations

import csv
import hashlib
import html
import inspect
import json
import logging
import os
import platform
import pickle
import sys
import threading
import time

# Set thread environment variables before torch import
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from collections import Counter
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
    BertTokenizerFast,
    DataCollatorWithPadding,
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
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(SCRIPT_DIR, "runs", RUN_ID)
LOG_DIR = os.path.join(RUN_DIR, "logs")
GRAPH_DIR = os.path.join(RUN_DIR, "graphs")
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "fine_tuned_bert_real")
REPORT_PATH = os.path.join(RUN_DIR, "training_report.md")
RESULTS_JSON_PATH = os.path.join(RUN_DIR, "training_results.json")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")

NUM_LABELS = 3
MAX_LENGTH = 128
TEST_SIZE = 0.10   # 10% test — keeps training set as large as possible
RANDOM_SEED = 42

# Training hyper-params (CPU-optimized)
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRADIENT_ACCUMULATION = 2  # effective batch = 32
LOGGING_STEPS = 100
EVAL_BATCH_SIZE = BATCH_SIZE * 4
DATALOADER_NUM_WORKERS = 2

LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %.2f", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def configure_torch_runtime(device: str) -> Dict[str, Any]:
    """Tune PyTorch threading for CPU training without changing model math."""

    if device != "cpu":
        return {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        }

    logical = psutil.cpu_count(logical=True) or 1
    threads = _env_int("BERT_CPU_THREADS", logical, minimum=1)
    interop_threads = _env_int("BERT_CPU_INTEROP_THREADS", 1, minimum=1)

    torch.set_num_threads(threads)
    torch.set_flush_denormal(True)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError as exc:
        logger.debug("Could not set inter-op threads: %s", exc)

    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True

    runtime_info = {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    if hasattr(torch.backends, "mkldnn"):
        runtime_info["mkldnn_enabled"] = bool(torch.backends.mkldnn.enabled)
    logger.info(
        "CPU runtime: torch threads=%s, inter-op=%s, mkldnn=%s",
        runtime_info["torch_num_threads"],
        runtime_info["torch_num_interop_threads"],
        runtime_info.get("mkldnn_enabled", "N/A"),
    )
    return runtime_info


def _token_cache_path() -> str:
    stat = os.stat(DATA_PATH)
    payload = {
        "dataset_size": stat.st_size,
        "dataset_mtime_ns": stat.st_mtime_ns,
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "padding": "dynamic",
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"bert_real_tokens_{digest}.pkl")


def tokenize_with_cache(
    tokenizer: BertTokenizerFast,
    train_texts: List[str],
    test_texts: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tokenize once, then reuse the same full-size split on future runs."""

    cache_path = _token_cache_path()
    use_cache = _env_bool("BERT_USE_TOKEN_CACHE", True)

    if use_cache and os.path.exists(cache_path):
        try:
            logger.info("Loading tokenized dataset from %s", cache_path)
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            return cached["train_enc"], cached["test_enc"]
        except (OSError, KeyError, pickle.PickleError) as exc:
            logger.warning("Token cache could not be read; rebuilding: %s", exc)

    logger.info("Tokenizing %d training samples ...", len(train_texts))
    train_enc = dict(tokenizer(
        train_texts,
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    ))
    logger.info("Tokenizing %d test samples ...", len(test_texts))
    test_enc = dict(tokenizer(
        test_texts,
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    ))

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {"train_enc": train_enc, "test_enc": test_enc},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info("Tokenized dataset cached at %s", cache_path)
        except OSError as exc:
            logger.warning("Could not write token cache: %s", exc)

    return train_enc, test_enc


def build_training_arguments(**kwargs: Any) -> TrainingArguments:
    """Only pass Trainer args supported by the installed transformers version."""

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    unsupported = sorted(k for k in kwargs if k not in params)
    if unsupported:
        logger.info("Skipping unsupported TrainingArguments: %s", ", ".join(unsupported))

    supported_kwargs = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**supported_kwargs)


# ---------------------------------------------------------------------------
# Energy & Resource Monitor
# ---------------------------------------------------------------------------


def _read_cpu_temperature_c() -> Tuple[Optional[float], Optional[str]]:
    """Return the hottest exposed temperature sensor, when supported."""

    if not hasattr(psutil, "sensors_temperatures"):
        return None, None

    try:
        sensors = psutil.sensors_temperatures(fahrenheit=False)
    except (AttributeError, OSError, RuntimeError):
        return None, None

    readings: List[Tuple[str, float]] = []
    for sensor_name, entries in sensors.items():
        for entry in entries:
            current = getattr(entry, "current", None)
            if current is None:
                continue
            label = getattr(entry, "label", "") or sensor_name
            readings.append((label.strip() or sensor_name, float(current)))

    if not readings:
        return None, None

    source, temperature = max(readings, key=lambda item: item[1])
    return temperature, source


class RaplReader:
    """Read Intel RAPL energy counters in joules from sysfs."""

    def __init__(self):
        self.path = "/sys/class/powercap/intel-rapl:0/energy_uj"
        self.available = os.path.exists(self.path)
        self.last_energy_j = 0.0

    def read_joules(self) -> Optional[float]:
        """Read current energy in joules. Returns None if unavailable."""
        if not self.available:
            return None
        try:
            with open(self.path, encoding="utf-8") as f:
                energy_uj = int(f.read().strip())
                return energy_uj / 1_000_000.0
        except (OSError, ValueError) as exc:
            logger.debug("RAPL read failed: %s", exc)
            return None


class EnergyMonitor:
    """Samples resource stats and estimates training energy consumption."""

    def __init__(self, sample_interval: float = 5.0, cpu_tdp_watts: float = 65.0):
        self.sample_interval = sample_interval
        self.cpu_tdp_watts = cpu_tdp_watts
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid())

        self.cpu_samples: List[float] = []
        self.process_cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.system_memory_percent_samples: List[float] = []
        self.temperature_samples: List[Optional[float]] = []
        self.cpu_freq_samples: List[Optional[float]] = []
        self.power_samples: List[float] = []
        self.rapl_energy_samples: List[float] = []
        self.energy_samples_wh: List[float] = []
        self.timestamps: List[float] = []
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self._last_sample_time: Optional[float] = None
        self._cumulative_energy_wh = 0.0
        self._temperature_source: Optional[str] = None
        self._start_cpu_times: Optional[Any] = None
        self._end_cpu_times: Optional[Any] = None
        
        # RAPL reader
        self.rapl = RaplReader()
        self.last_rapl_energy_j = 0.0
        
        # CSV logging
        self.csv_path = os.path.join(LOG_DIR, "system_monitor.csv")
        self.csv_file: Optional[Any] = None
        self.csv_writer: Optional[Any] = None

    def start(self) -> None:
        self.start_time = time.time()
        self._running = True
        self._process = psutil.Process(os.getpid())
        self._start_cpu_times = self._process.cpu_times()
        psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)
        
        # Initialize RAPL energy baseline
        if self.rapl.available:
            initial_rapl = self.rapl.read_joules()
            if initial_rapl is not None:
                self.last_rapl_energy_j = initial_rapl
        
        # Initialize CSV logging
        try:
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "timestamp_s", "system_cpu_percent", "process_cpu_percent",
                "process_memory_mb", "temperature_c", "cpu_freq_mhz",
                "rapl_energy_j", "estimated_power_w"
            ])
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())
        except OSError as exc:
            logger.warning("Could not open CSV log: %s", exc)
        
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info("Energy monitor started (sampling every %.1fs)", self.sample_interval)
        logger.info("RAPL available: %s", self.rapl.available)

    def stop(self) -> Dict[str, Any]:
        self._running = False
        self.end_time = time.time()
        if self._thread:
            self._thread.join(timeout=10)
        if self._last_sample_time and self.power_samples:
            trailing_h = max(0.0, (self.end_time - self._last_sample_time) / 3600.0)
            self._cumulative_energy_wh += self.power_samples[-1] * trailing_h
            self.energy_samples_wh[-1] = self._cumulative_energy_wh
        
        # Close CSV file
        if self.csv_file:
            try:
                self.csv_file.close()
                logger.info("System monitor CSV saved to %s", self.csv_path)
            except OSError as exc:
                logger.warning("Could not close CSV file: %s", exc)
        
        try:
            self._end_cpu_times = self._process.cpu_times()
        except (psutil.Error, OSError):
            self._end_cpu_times = None
        return self.get_report()

    def _sample_loop(self) -> None:
        while self._running:
            try:
                now = time.time()
                elapsed_s = now - self.start_time
                cpu = psutil.cpu_percent(interval=None)
                process_cpu = self._process.cpu_percent(interval=None)
                mem_info = self._process.memory_info()
                mem = mem_info.rss / (1024 * 1024)
                system_mem = psutil.virtual_memory()
                temp_c, temp_source = _read_cpu_temperature_c()
                
                # Read CPU frequency
                cpu_freq = psutil.cpu_freq()
                cpu_freq_mhz = cpu_freq.current if cpu_freq else None
                
                # Calculate power from RAPL
                rapl_energy_j = 0.0
                power_w = 0.0
                if self.rapl.available:
                    current_rapl = self.rapl.read_joules()
                    if current_rapl is not None:
                        rapl_energy_j = current_rapl
                        if self._last_sample_time is not None:
                            elapsed_h = max(0.0, (now - self._last_sample_time) / 3600.0)
                            delta_j = current_rapl - self.last_rapl_energy_j
                            if elapsed_h > 0:
                                power_w = delta_j / (elapsed_h * 3600.0)  # W = J / seconds
                            self._cumulative_energy_wh += power_w * elapsed_h
                            self.last_rapl_energy_j = current_rapl
                        else:
                            self.last_rapl_energy_j = current_rapl

                self._last_sample_time = now

                self.cpu_samples.append(cpu)
                self.process_cpu_samples.append(process_cpu)
                self.memory_samples.append(mem)
                self.system_memory_percent_samples.append(system_mem.percent)
                self.temperature_samples.append(temp_c)
                self.cpu_freq_samples.append(cpu_freq_mhz)
                self.power_samples.append(power_w)
                self.rapl_energy_samples.append(rapl_energy_j)
                self.energy_samples_wh.append(self._cumulative_energy_wh)
                self.timestamps.append(now)

                if temp_c is not None and self._temperature_source is None:
                    self._temperature_source = temp_source
                
                # Write to CSV
                if self.csv_writer and self.csv_file:
                    try:
                        self.csv_writer.writerow([
                            round(elapsed_s, 1),
                            round(cpu, 2),
                            round(process_cpu, 2),
                            round(mem, 2),
                            round(temp_c, 2) if temp_c is not None else "",
                            round(cpu_freq_mhz, 0) if cpu_freq_mhz is not None else "",
                            round(rapl_energy_j, 2),
                            round(power_w, 2),
                        ])
                        self.csv_file.flush()
                        os.fsync(self.csv_file.fileno())
                    except (OSError, IOError) as exc:
                        logger.debug("CSV write failed: %s", exc)
            except (psutil.Error, OSError, RuntimeError) as exc:
                logger.debug("Energy monitor sample skipped: %s", exc)
            time.sleep(self.sample_interval)

    def get_report(self) -> Dict[str, Any]:
        duration_s = self.end_time - self.start_time
        duration_h = duration_s / 3600.0

        if not self.cpu_samples:
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            return {
                "error": "No samples collected",
                "duration_seconds": round(duration_s, 1),
                "duration_formatted": str(timedelta(seconds=int(duration_s))),
                "num_samples_collected": 0,
                "cpu": {
                    "num_cores": psutil.cpu_count(logical=True),
                    "physical_cores": psutil.cpu_count(logical=False),
                },
                "memory": {"total_system_gb": round(total_ram_gb, 1)},
                "temperature": {"available": False},
                "samples": {},
                "monitoring": {
                    "sample_interval_seconds": self.sample_interval,
                    "energy_method": "No samples collected.",
                },
            }

        avg_cpu = float(np.mean(self.cpu_samples))
        max_cpu = float(np.max(self.cpu_samples))
        min_cpu = float(np.min(self.cpu_samples))
        avg_process_cpu = float(np.mean(self.process_cpu_samples))
        max_process_cpu = float(np.max(self.process_cpu_samples))

        avg_mem = float(np.mean(self.memory_samples))
        max_mem = float(np.max(self.memory_samples))
        avg_system_mem = float(np.mean(self.system_memory_percent_samples))
        max_system_mem = float(np.max(self.system_memory_percent_samples))

        avg_power_w = float(np.mean(self.power_samples)) if self.power_samples else 0.0
        max_power_w = float(np.max(self.power_samples)) if self.power_samples else 0.0

        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        energy_wh = (
            self._cumulative_energy_wh
            if self._cumulative_energy_wh > 0
            else avg_power_w * duration_h
        )
        energy_kwh = energy_wh / 1000.0
        co2_grams = energy_kwh * 475.0
        co2_grams_india = energy_kwh * 720.0

        process_cpu = self._process_cpu_report(duration_s)
        temp_values = [t for t in self.temperature_samples if t is not None]
        relative_times = [round(t - self.start_time, 1) for t in self.timestamps]

        return {
            "duration_seconds": round(duration_s, 1),
            "duration_formatted": str(timedelta(seconds=int(duration_s))),
            "num_samples_collected": len(self.cpu_samples),
            "cpu": {
                "average_percent": round(avg_cpu, 1),
                "max_percent": round(max_cpu, 1),
                "min_percent": round(min_cpu, 1),
                "process_average_percent": round(avg_process_cpu, 1),
                "process_max_percent": round(max_process_cpu, 1),
                "num_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                **process_cpu,
            },
            "memory": {
                "average_mb": round(avg_mem, 1),
                "peak_mb": round(max_mem, 1),
                "average_process_mb": round(avg_mem, 1),
                "peak_process_mb": round(max_mem, 1),
                "average_system_percent": round(avg_system_mem, 1),
                "peak_system_percent": round(max_system_mem, 1),
                "total_system_gb": round(total_ram_gb, 1),
            },
            "temperature": {
                "available": bool(temp_values),
                "source": self._temperature_source,
                "average_c": round(float(np.mean(temp_values)), 1) if temp_values else None,
                "peak_c": round(float(np.max(temp_values)), 1) if temp_values else None,
                "min_c": round(float(np.min(temp_values)), 1) if temp_values else None,
                "num_samples": len(temp_values),
            },
            "energy": {
                "rapl_available": self.rapl.available,
                "estimated_avg_power_watts": round(avg_power_w, 1),
                "estimated_peak_power_watts": round(max_power_w, 1),
                "total_energy_wh": round(energy_wh, 2),
                "total_energy_kwh": round(energy_kwh, 4),
                "measurement_method": "Intel RAPL" if self.rapl.available else "Estimated from CPU load and TDP",
            },
            "carbon": {
                "co2_grams_global_avg": round(co2_grams, 2),
                "co2_grams_india_avg": round(co2_grams_india, 2),
            },
            "samples": {
                "time_seconds": relative_times,
                "system_cpu_percent": [round(v, 2) for v in self.cpu_samples],
                "process_cpu_percent": [round(v, 2) for v in self.process_cpu_samples],
                "process_memory_mb": [round(v, 2) for v in self.memory_samples],
                "system_memory_percent": [
                    round(v, 2) for v in self.system_memory_percent_samples
                ],
                "temperature_c": [
                    round(v, 2) if v is not None else None
                    for v in self.temperature_samples
                ],
                "cpu_freq_mhz": [
                    round(v, 0) if v is not None else None
                    for v in self.cpu_freq_samples
                ],
                "estimated_power_watts": [round(v, 2) for v in self.power_samples],
                "rapl_energy_j": [round(v, 2) for v in self.rapl_energy_samples],
                "cumulative_energy_wh": [
                    round(v, 4) for v in self.energy_samples_wh
                ],
            },
            "monitoring": {
                "sample_interval_seconds": self.sample_interval,
                "temperature_available": bool(temp_values),
                "temperature_source": self._temperature_source,
                "rapl_available": self.rapl.available,
                "energy_method": (
                    "Intel RAPL (Running Average Power Limit) energy counters from /sys/class/powercap/"
                    if self.rapl.available
                    else "Estimated from sampled system CPU utilization and configured CPU TDP"
                ),
            },
        }

    def _process_cpu_report(self, duration_s: float) -> Dict[str, Any]:
        if not self._start_cpu_times or not self._end_cpu_times:
            return {}

        user_s = max(0.0, self._end_cpu_times.user - self._start_cpu_times.user)
        system_s = max(0.0, self._end_cpu_times.system - self._start_cpu_times.system)
        total_s = user_s + system_s
        avg_cores = total_s / duration_s if duration_s > 0 else 0.0
        logical = psutil.cpu_count(logical=True) or 1
        avg_process_share = (avg_cores / logical) * 100.0

        return {
            "process_user_seconds": round(user_s, 2),
            "process_system_seconds": round(system_s, 2),
            "process_total_cpu_seconds": round(total_s, 2),
            "process_cpu_time_formatted": str(timedelta(seconds=int(total_s))),
            "average_process_cores_used": round(avg_cores, 2),
            "average_process_cpu_share_percent": round(avg_process_share, 1),
        }


# ---------------------------------------------------------------------------
# Training metrics callback
# ---------------------------------------------------------------------------


class MetricsLogger(TrainerCallback):
    """Logs per-step and per-epoch metrics for the training report."""

    def __init__(self):
        self.step_losses: List[Dict[str, Any]] = []
        self.epoch_metrics: List[Dict[str, Any]] = []
        
        # CSV logging
        self.loss_csv_path = os.path.join(LOG_DIR, "training_loss.csv")
        self.epoch_csv_path = os.path.join(LOG_DIR, "epoch_metrics.csv")
        self.loss_csv_file: Optional[Any] = None
        self.loss_csv_writer: Optional[Any] = None
        self.epoch_csv_file: Optional[Any] = None
        self.epoch_csv_writer: Optional[Any] = None
        
        # Initialize CSV files
        self._init_csv_files()

    def _init_csv_files(self) -> None:
        """Initialize CSV files for logging."""
        try:
            self.loss_csv_file = open(self.loss_csv_path, "w", newline="", encoding="utf-8")
            self.loss_csv_writer = csv.writer(self.loss_csv_file)
            self.loss_csv_writer.writerow(["step", "epoch", "loss", "learning_rate", "timestamp"])
            self.loss_csv_file.flush()
            os.fsync(self.loss_csv_file.fileno())
        except OSError as exc:
            logger.warning("Could not open loss CSV: %s", exc)
        
        try:
            self.epoch_csv_file = open(self.epoch_csv_path, "w", newline="", encoding="utf-8")
            self.epoch_csv_writer = csv.writer(self.epoch_csv_file)
            self.epoch_csv_writer.writerow(["epoch", "accuracy", "f1", "precision", "recall", "eval_loss"])
            self.epoch_csv_file.flush()
            os.fsync(self.epoch_csv_file.fileno())
        except OSError as exc:
            logger.warning("Could not open epoch CSV: %s", exc)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 2)}
            entry.update({k: v for k, v in logs.items()
                        if isinstance(v, (int, float))})
            self.step_losses.append(entry)
            
            # Log to CSV if this is a loss entry
            if "loss" in entry and self.loss_csv_writer:
                try:
                    self.loss_csv_writer.writerow([
                        entry["step"],
                        entry["epoch"],
                        round(entry["loss"], 4),
                        round(entry.get("learning_rate", 0), 6),
                        datetime.now().isoformat(),
                    ])
                    self.loss_csv_file.flush()
                    os.fsync(self.loss_csv_file.fileno())
                except (OSError, IOError) as exc:
                    logger.debug("Loss CSV write failed: %s", exc)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 2)}
            entry.update({k: round(v, 4) if isinstance(v, float) else v
                        for k, v in metrics.items()})
            self.epoch_metrics.append(entry)
            
            # Log to CSV
            if self.epoch_csv_writer:
                try:
                    self.epoch_csv_writer.writerow([
                        entry["epoch"],
                        entry.get("eval_accuracy", ""),
                        entry.get("eval_f1_macro", ""),
                        entry.get("eval_precision_macro", ""),
                        entry.get("eval_recall_macro", ""),
                        entry.get("eval_loss", ""),
                    ])
                    self.epoch_csv_file.flush()
                    os.fsync(self.epoch_csv_file.fileno())
                except (OSError, IOError) as exc:
                    logger.debug("Epoch CSV write failed: %s", exc)

    def close_csv_files(self) -> None:
        """Close CSV files."""
        if self.loss_csv_file:
            try:
                self.loss_csv_file.close()
                logger.info("Training loss CSV saved to %s", self.loss_csv_path)
            except OSError as exc:
                logger.warning("Could not close loss CSV: %s", exc)
        if self.epoch_csv_file:
            try:
                self.epoch_csv_file.close()
                logger.info("Epoch metrics CSV saved to %s", self.epoch_csv_path)
            except OSError as exc:
                logger.warning("Could not close epoch CSV: %s", exc)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class SentimentDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.lengths = [len(ids) for ids in encodings.get("input_ids", [])]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
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
# Report graph helpers
# ---------------------------------------------------------------------------


def _format_metric(value: Any, decimals: int = 1, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if np.isnan(number):
            return "N/A"
        return f"{number:,.{decimals}f}{suffix}"
    return str(value)


def _relative_report_path(path: str) -> str:
    return os.path.relpath(path, SCRIPT_DIR).replace(os.sep, "/")


def _write_line_chart(
    output_path: str,
    title: str,
    y_label: str,
    series: List[Dict[str, Any]],
    x_label: str = "Elapsed time (minutes)",
    subtitle: str = "",
) -> bool:
    cleaned = []
    for item in series:
        points = [
            (float(x), float(y))
            for x, y in zip(item.get("x", []), item.get("y", []))
            if y is not None
        ]
        if points:
            cleaned.append({**item, "points": points})

    if not cleaned:
        return False

    width, height = 980, 360
    left, right, top, bottom = 78, 28, 58, 62
    chart_w = width - left - right
    chart_h = height - top - bottom

    all_x = [x for item in cleaned for x, _ in item["points"]]
    all_y = [y for item in cleaned for _, y in item["points"]]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        pad = max(abs(y_min) * 0.1, 1.0)
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.12
        y_min -= pad
        y_max += pad

    def sx(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * chart_w

    def sy(value: float) -> float:
        return top + chart_h - ((value - y_min) / (y_max - y_min)) * chart_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{left}" y="26" font-family="Arial, sans-serif" '
            f'font-size="18" font-weight="700" fill="#111827">'
            f'{html.escape(title)}</text>'
        ),
    ]
    if subtitle:
        elements.append(
            f'<text x="{left}" y="45" font-family="Arial, sans-serif" '
            f'font-size="12" fill="#4b5563">{html.escape(subtitle)}</text>'
        )

    for i in range(6):
        y_value = y_min + ((y_max - y_min) * i / 5)
        y = sy(y_value)
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_w}" '
            f'y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#6b7280">'
            f'{_format_metric(y_value, 1)}</text>'
        )

    for i in range(6):
        x_value = x_min + ((x_max - x_min) * i / 5)
        x = sx(x_value)
        elements.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{top + chart_h}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{top + chart_h + 24}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#6b7280">'
            f'{_format_metric(x_value, 1)}</text>'
        )

    elements.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" '
        'stroke="#9ca3af" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" '
        f'y2="{top + chart_h}" stroke="#9ca3af" stroke-width="1.2"/>',
        (
            f'<text x="{left + chart_w / 2:.2f}" y="{height - 14}" '
            f'text-anchor="middle" font-family="Arial, sans-serif" '
            f'font-size="12" fill="#374151">{html.escape(x_label)}</text>'
        ),
        (
            f'<text x="18" y="{top + chart_h / 2:.2f}" text-anchor="middle" '
            f'transform="rotate(-90 18 {top + chart_h / 2:.2f})" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#374151">'
            f'{html.escape(y_label)}</text>'
        ),
    ])

    legend_x = left + chart_w - 250
    legend_y = 24
    for idx, item in enumerate(cleaned):
        color = item.get("color", "#2563eb")
        label = html.escape(str(item.get("label", f"Series {idx + 1}")))
        y = legend_y + idx * 18
        elements.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" '
            f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        )
        elements.append(
            f'<text x="{legend_x + 28}" y="{y + 4}" font-family="Arial, sans-serif" '
            f'font-size="11" fill="#374151">{label}</text>'
        )

    for item in cleaned:
        color = item.get("color", "#2563eb")
        points = " ".join(
            f"{sx(x):.2f},{sy(y):.2f}" for x, y in item["points"]
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        end_x, end_y = item["points"][-1]
        elements.append(
            f'<circle cx="{sx(end_x):.2f}" cy="{sy(end_y):.2f}" r="3.4" '
            f'fill="{color}"/>'
        )

    elements.append("</svg>")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(elements))
    return True


def generate_monitoring_graphs(
    energy_report: Dict[str, Any],
    metrics_logger: MetricsLogger,
    graph_dir: str,
) -> List[Dict[str, str]]:
    graphs: List[Dict[str, str]] = []
    samples = energy_report.get("samples", {})
    time_min = [t / 60.0 for t in samples.get("time_seconds", [])]
    logical_cores = energy_report.get("cpu", {}).get("num_cores") or 1

    def add_graph(filename: str, title: str, description: str, **kwargs: Any) -> None:
        path = os.path.join(graph_dir, filename)
        if _write_line_chart(output_path=path, title=title, **kwargs):
            graphs.append({
                "title": title,
                "path": _relative_report_path(path),
                "description": description,
            })

    if time_min:
        process_share = [
            min(100.0, value / logical_cores)
            for value in samples.get("process_cpu_percent", [])
        ]
        add_graph(
            "cpu_usage.svg",
            "CPU Utilization During Run",
            "System CPU load and the training process share normalized across all logical cores.",
            y_label="CPU utilization (%)",
            subtitle="Higher sustained values mean the CPU is the dominant bottleneck.",
            series=[
                {
                    "label": "System CPU",
                    "x": time_min,
                    "y": samples.get("system_cpu_percent", []),
                    "color": "#2563eb",
                },
                {
                    "label": "Training process CPU share",
                    "x": time_min,
                    "y": process_share,
                    "color": "#dc2626",
                },
            ],
        )

        add_graph(
            "memory_usage.svg",
            "Memory Usage During Run",
            "Resident memory used by the training Python process.",
            y_label="Process memory (MB)",
            subtitle="Peak memory is useful for checking whether the run fits non-GPU machines.",
            series=[{
                "label": "Process RSS",
                "x": time_min,
                "y": samples.get("process_memory_mb", []),
                "color": "#059669",
            }],
        )

        add_graph(
            "power_estimate.svg",
            "Estimated Power Draw",
            "Estimated instantaneous power from CPU load and process memory footprint.",
            y_label="Estimated watts",
            subtitle="This is a software estimate, not a hardware wattmeter reading.",
            series=[{
                "label": "Estimated power",
                "x": time_min,
                "y": samples.get("estimated_power_watts", []),
                "color": "#7c3aed",
            }],
        )

        add_graph(
            "energy_accumulation.svg",
            "Cumulative Estimated Energy",
            "Estimated energy accumulated over the monitored run.",
            y_label="Energy (Wh)",
            subtitle="The final point should align with the report's total estimated energy.",
            series=[{
                "label": "Cumulative energy",
                "x": time_min,
                "y": samples.get("cumulative_energy_wh", []),
                "color": "#ea580c",
            }],
        )

        temperature = samples.get("temperature_c", [])
        if any(value is not None for value in temperature):
            add_graph(
                "temperature.svg",
                "System Temperature",
                "Highest exposed CPU/system temperature sensor sampled during the run.",
                y_label="Temperature (C)",
                subtitle="Temperature appears only when the OS exposes sensor readings.",
                series=[{
                    "label": "Temperature",
                    "x": time_min,
                    "y": temperature,
                    "color": "#b91c1c",
                }],
            )

    loss_entries = [s for s in metrics_logger.step_losses if "loss" in s]
    if loss_entries:
        add_graph(
            "training_loss.svg",
            "Training Loss Progression",
            "Logged training loss across optimizer steps.",
            x_label="Optimizer step",
            y_label="Loss",
            subtitle="A downward trend usually indicates the model is learning.",
            series=[{
                "label": "Training loss",
                "x": [entry["step"] for entry in loss_entries],
                "y": [entry["loss"] for entry in loss_entries],
                "color": "#2563eb",
            }],
        )

    epoch_metrics = [
        entry for entry in metrics_logger.epoch_metrics
        if "eval_f1_macro" in entry or "eval_accuracy" in entry
    ]
    if epoch_metrics:
        add_graph(
            "evaluation_metrics.svg",
            "Evaluation Metrics Per Epoch",
            "Held-out test metrics recorded after each epoch.",
            x_label="Epoch",
            y_label="Score",
            subtitle="F1 macro is the selected best-model metric.",
            series=[
                {
                    "label": "F1 macro",
                    "x": [entry["epoch"] for entry in epoch_metrics],
                    "y": [entry.get("eval_f1_macro") for entry in epoch_metrics],
                    "color": "#7c3aed",
                },
                {
                    "label": "Accuracy",
                    "x": [entry["epoch"] for entry in epoch_metrics],
                    "y": [entry.get("eval_accuracy") for entry in epoch_metrics],
                    "color": "#059669",
                },
            ],
        )

    return graphs


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
    graph_info: Optional[List[Dict[str, str]]] = None,
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
    temp_info = energy_report.get("temperature", {})
    carbon = energy_report.get("carbon", {})
    monitoring_info = energy_report.get("monitoring", {})
    graph_info = graph_info or []

    graph_section = "No monitoring graphs were generated because no telemetry samples were collected.\n"
    if graph_info:
        graph_parts = []
        for graph in graph_info:
            graph_parts.append(
                f"#### {graph['title']}\n\n"
                f"{graph['description']}\n\n"
                f"![{graph['title']}]({graph['path']})\n"
            )
        graph_section = "\n".join(graph_parts)

    temperature_text = (
        f"{_format_metric(temp_info.get('average_c'), 1, ' C')} average, "
        f"{_format_metric(temp_info.get('peak_c'), 1, ' C')} peak "
        f"from `{temp_info.get('source')}`"
        if temp_info.get("available")
        else "Not available from the operating system sensor API on this machine."
    )

    report = f"""# BERT Fine-Tuning Training Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model**: `{MODEL_NAME}` -> 3-class sentiment (Negative / Neutral / Positive)

This report is generated at the end of the run from the same telemetry samples
collected by the training process. Energy is reported as a software estimate;
CPU, memory, process CPU time, elapsed time, and temperature are sampled with
`psutil` where the operating system exposes those values.

---

## 1. System Information

| Parameter | Value |
|-----------|-------|
| OS | {system_info.get('os', 'N/A')} |
| CPU | {system_info.get('cpu', 'N/A')} |
| CPU Cores (physical/logical) | {cpu_info.get('physical_cores', '?')}/{cpu_info.get('num_cores', '?')} |
| Total RAM | {mem_info.get('total_system_gb', '?')} GB |
| CPU Frequency | {system_info.get('cpu_frequency', 'N/A')} |
| Machine | {system_info.get('machine', 'N/A')} |
| Python | {system_info.get('python', 'N/A')} |
| PyTorch | {system_info.get('pytorch', 'N/A')} |
| Device | {system_info.get('device', 'CPU')} |
| GPU | {'None (CPU-only training)' if system_info.get('device') == 'cpu' else system_info.get('gpu', 'N/A')} |
| Torch Threads | {system_info.get('torch_num_threads', 'N/A')} |
| Torch Inter-op Threads | {system_info.get('torch_num_interop_threads', 'N/A')} |

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
| Padding | Dynamic per batch, truncated at max length |
| Epochs | {EPOCHS} |
| Batch Size (per device) | {BATCH_SIZE} |
| Eval Batch Size (per device) | {system_info.get('eval_batch_size', BATCH_SIZE * 4)} |
| Gradient Accumulation Steps | {GRADIENT_ACCUMULATION} |
| Effective Batch Size | {BATCH_SIZE * GRADIENT_ACCUMULATION} |
| Learning Rate | {LEARNING_RATE} |
| Weight Decay | {WEIGHT_DECAY} |
| Warmup Ratio | {WARMUP_RATIO} |
| DataLoader Workers | {system_info.get('dataloader_workers', 0)} |
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
| **Monitored Run Time** | **{energy_report.get('duration_formatted', 'N/A')}** |
| Monitored Seconds | {energy_report.get('duration_seconds', 'N/A')} |
| Fine-Tuning `trainer.train()` Time | {str(timedelta(seconds=int(training_time_s)))} |
| Process CPU Time | {cpu_info.get('process_cpu_time_formatted', 'N/A')} |
| Process CPU User/System Time | {_format_metric(cpu_info.get('process_user_seconds'), 2, 's')} / {_format_metric(cpu_info.get('process_system_seconds'), 2, 's')} |
| Samples/Second | {dataset_stats['train_size'] * EPOCHS / training_time_s:.1f} |
| Steps/Second | {train_result.metrics.get('train_steps_per_second', 'N/A')} |

---

## 6. Energy, Resources, And Temperature

### Monitoring Coverage

| Metric | Value |
|--------|-------|
| Sampling Interval | {_format_metric(monitoring_info.get('sample_interval_seconds'), 1, 's')} |
| Monitoring Samples | {energy_report.get('num_samples_collected', '?')} |
| Temperature Sensor | {temperature_text} |
| Energy Measurement Method | {monitoring_info.get('energy_method', 'N/A')} |

### Resource Utilization

| Metric | Value |
|--------|-------|
| Avg System CPU Usage | {_format_metric(cpu_info.get('average_percent'), 1, '%')} |
| Peak System CPU Usage | {_format_metric(cpu_info.get('max_percent'), 1, '%')} |
| Avg Training Process CPU | {_format_metric(cpu_info.get('process_average_percent'), 1, '%')} |
| Peak Training Process CPU | {_format_metric(cpu_info.get('process_max_percent'), 1, '%')} |
| Avg Process Cores Used | {_format_metric(cpu_info.get('average_process_cores_used'), 2)} |
| Avg Process CPU Share | {_format_metric(cpu_info.get('average_process_cpu_share_percent'), 1, '%')} |
| Avg Process Memory | {_format_metric(mem_info.get('average_process_mb'), 0, ' MB')} |
| Peak Process Memory | {_format_metric(mem_info.get('peak_process_mb'), 0, ' MB')} |
| Avg System Memory Usage | {_format_metric(mem_info.get('average_system_percent'), 1, '%')} |
| Peak System Memory Usage | {_format_metric(mem_info.get('peak_system_percent'), 1, '%')} |
| Avg Temperature | {_format_metric(temp_info.get('average_c'), 1, ' C')} |
| Peak Temperature | {_format_metric(temp_info.get('peak_c'), 1, ' C')} |

### Energy Estimates

| Metric | Value |
|--------|-------|
| Estimated CPU TDP | {en.get('estimated_cpu_tdp_watts', '?')} W |
| Estimated Avg CPU Power | {en.get('estimated_cpu_power_watts', '?')} W |
| Estimated RAM Power | {en.get('estimated_ram_power_watts', '?')} W |
| Estimated Peak Power | {en.get('estimated_peak_power_watts', '?')} W |
| **Estimated Total Avg Power** | **{en.get('estimated_avg_power_watts', '?')} W** |
| **Total Energy Consumed** | **{en.get('total_energy_wh', '?')} Wh ({en.get('total_energy_kwh', '?')} kWh)** |

### Carbon Footprint

| Region | CO2 Emissions |
|--------|--------------|
| Global Average (475g CO2/kWh) | {carbon.get('co2_grams_global_avg', '?')} g CO2 |
| India Average (720g CO2/kWh) | {carbon.get('co2_grams_india_avg', '?')} g CO2 |

### Monitoring Graphs

{graph_section}

> Annotation: energy is estimated from sampled CPU load, configured CPU TDP,
> and process memory usage. Temperature is reported only when the OS exposes
> sensor data through `psutil`; otherwise the report explicitly marks it as
> unavailable instead of inventing a value.

---

## 7. Model Artifacts

| File | Description |
|------|-------------|
| `fine_tuned_bert_real/` | Saved model weights & tokenizer |
| `fine_tuned_bert_real/report_assets/` | SVG graphs embedded in this report |
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
    print("=" * 70)
    print("  BERT FINE-TUNING ON FULL REAL-WORLD DATASET")
    print("  with Energy Monitoring & Training Report")
    print("=" * 70)

    # ---- System info ----
    cpu_freq = psutil.cpu_freq()
    system_info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor() or "Unknown",
        "machine": platform.machine(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "cpu_frequency": (
            f"{cpu_freq.current:.0f} MHz current / {cpu_freq.max:.0f} MHz max"
            if cpu_freq else "N/A"
        ),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        system_info["gpu"] = torch.cuda.get_device_name(0)

    device = system_info["device"]
    logger.info("Device: %s", device)
    system_info.update(configure_torch_runtime(device))

    eval_batch_size = _env_int(
        "BERT_EVAL_BATCH_SIZE", EVAL_BATCH_SIZE, minimum=1
    )
    dataloader_num_workers = _env_int(
        "BERT_DATALOADER_WORKERS", DATALOADER_NUM_WORKERS, minimum=0
    )
    system_info["eval_batch_size"] = eval_batch_size
    system_info["dataloader_workers"] = dataloader_num_workers

    # ---- 0. Create run directories ----
    logger.info("Creating run directories: %s", RUN_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---- 1. Load data ----
    logger.info("Loading dataset from %s ...", DATA_PATH)
    if not os.path.exists(DATA_PATH):
        logger.error("Dataset not found! Run: python download_real_data.py")
        sys.exit(1)

    energy_monitor = EnergyMonitor(
        sample_interval=_env_float("BERT_MONITOR_INTERVAL_SECONDS", 5.0, minimum=0.5),
        cpu_tdp_watts=_env_float("BERT_CPU_TDP_WATTS", 65.0, minimum=1.0),
    )
    energy_monitor.start()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    texts = [s["text"] for s in raw_data]
    labels = [s["label"] for s in raw_data]
    sources = [s.get("source", "unknown") for s in raw_data]

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
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    train_enc, test_enc = tokenize_with_cache(tokenizer, train_texts, test_texts)

    train_dataset = SentimentDataset(train_enc, train_labels)
    test_dataset = SentimentDataset(test_enc, test_labels)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Compute step counts here so warmup_steps can be passed to TrainingArguments
    effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION
    steps_per_epoch = (len(train_dataset) + effective_batch - 1) // effective_batch
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    # ---- 4. Load model ----
    logger.info("Loading model: %s (num_labels=%d)", MODEL_NAME, NUM_LABELS)
    # Silence the verbose UNEXPECTED/MISSING key report and HF Hub auth warning —
    # both are expected and harmless for fine-tuning (see comments below).
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    # Restore transformers logger to INFO for the rest of training
    logging.getLogger("transformers.modeling_utils").setLevel(logging.INFO)

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
    # Set TensorBoard log dir via env var (replaces deprecated logging_dir arg)
    os.environ["TENSORBOARD_LOGGING_DIR"] = LOG_DIR

    training_args = build_training_arguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,          # replaces deprecated warmup_ratio
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=5,
        save_only_model=True,
        report_to="none",
        fp16=False,
        optim="adamw_torch",
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_num_workers=dataloader_num_workers,
        dataloader_persistent_workers=dataloader_num_workers > 0,
        dataloader_prefetch_factor=2 if dataloader_num_workers > 0 else None,
        group_by_length=True,
        seed=RANDOM_SEED,
        disable_tqdm=False,
    )

    # ---- 7. Custom Trainer with weighted loss ----
    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_t = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if self.class_weights.device != logits.device:
                self.class_weights = self.class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels_t,
                weight=self.class_weights,
            )
            return (loss, outputs) if return_outputs else loss

    metrics_logger = MetricsLogger()

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[metrics_logger],
        class_weights=weights,
    )

    # ---- 8. Train ----
    eval_steps = steps_per_epoch  # eval_strategy="epoch" → eval after every epoch

    logger.info("=" * 60)
    logger.info("  STARTING TRAINING on %d samples, %d epochs", len(train_texts), EPOCHS)
    logger.info("  Optimizer steps per epoch: %d", steps_per_epoch)
    logger.info("  Eval steps (per epoch): %d", eval_steps)
    logger.info("=" * 60)

    # Only resume if a checkpoint actually exists in the output directory
    checkpoint_dirs = [
        d for d in os.listdir(CHECKPOINT_DIR)
        if d.startswith("checkpoint-")
    ] if os.path.isdir(CHECKPOINT_DIR) else []
    resume = checkpoint_dirs[-1] and os.path.join(CHECKPOINT_DIR, sorted(checkpoint_dirs)[-1]) if checkpoint_dirs else None

    train_start = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume)
    training_time_s = time.time() - train_start

    logger.info("Training completed in %s", str(timedelta(seconds=int(training_time_s))))
    
    # Close metrics logger CSV files
    metrics_logger.close_csv_files()

    # ---- 9. Final prediction/evaluation ----
    logger.info("Running final prediction/evaluation on test set ...")
    preds_output = trainer.predict(test_dataset, metric_key_prefix="eval")
    eval_metrics = preds_output.metrics
    logger.info("Eval: %s", eval_metrics)

    test_preds = np.argmax(preds_output.predictions, axis=-1)
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(NUM_LABELS)))

    # ---- 10. Save model ----
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
    meta_path = os.path.join(RUN_DIR, "test_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(test_meta, f, indent=2, ensure_ascii=False)

    # ---- 11. Stop monitor and build graph assets ----
    energy_report = energy_monitor.stop()
    en = energy_report.get("energy", {})
    carbon = energy_report.get("carbon", {})
    temp = energy_report.get("temperature", {})
    graph_info = generate_monitoring_graphs(energy_report, metrics_logger, GRAPH_DIR)

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
        graph_info=graph_info,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Report saved to %s", REPORT_PATH)

    # Save raw results JSON
    results_json = {
        "system_info": system_info,
        "dataset_stats": dict(dataset_stats),
        "training_config": {
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": eval_batch_size,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "padding": "dynamic",
            "dataloader_workers": dataloader_num_workers,
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
        "monitoring_graphs": graph_info,
        "epoch_metrics": metrics_logger.epoch_metrics,
        "confusion_matrix": cm.tolist(),
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
    print(f"  CPU time     : {energy_report.get('cpu', {}).get('process_cpu_time_formatted', 'N/A')}")
    print(f"  Peak memory  : {energy_report.get('memory', {}).get('peak_process_mb', 'N/A')} MB")
    print(f"  Peak temp    : {temp.get('peak_c', 'N/A')} C")
    print(f"  CO2 (India)  : {carbon.get('co2_grams_india_avg', '?')} g")
    print(f"  Report       : {REPORT_PATH}")
    print(f"  Graphs       : {GRAPH_DIR}")
    print(f"  Model        : {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()