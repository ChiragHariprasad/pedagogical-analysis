"""
BERT Sentiment Inference Pipeline
==================================
Inference engine for the fine-tuned bert-base-uncased 3-class sentiment model
produced by finetune_bert_real.py.

Handles three concerns in one file:

  1. EXPORT    -- Convert fine-tuned PyTorch model -> ONNX -> OpenVINO IR
                  with optional INT8 post-training quantization

  2. BENCH     -- Benchmark raw PyTorch vs ONNX vs OpenVINO on your hardware
                  and recommend the fastest backend

  3. INFER     -- SentimentInferenceEngine class and CLI for single, batch,
                  and interactive inference

Compatible with finetune_bert_real.py output layout:
    fine_tuned_bert_real/          <- model weights + tokenizer (default)
    bert_real_onnx/model.onnx      <- ONNX export  (created by --export)
    bert_real_openvino/model.xml   <- OpenVINO IR   (created by --export)

Label schema (must match finetune_bert_real.py):
    0 -> Negative
    1 -> Neutral
    2 -> Positive

Hardware targets:
  * Intel NPU     -- OpenVINO NPU plugin (Intel Core Ultra / Meteor Lake+)
  * Intel CPU     -- OpenVINO CPU plugin (always available on Intel)
  * Fallback      -- Raw PyTorch on CPU  (no ONNX/OpenVINO required)

Usage:
    # 1. Export + quantize (run once after finetune_bert_real.py completes)
    python run_inference_bert_real.py --export

    # 2. Benchmark all backends on your hardware
    python run_inference_bert_real.py --benchmark

    # 3. Score a single sentence
    python run_inference_bert_real.py --test "The lecture was incredibly well-structured"

    # 4. Interactive mode (default)
    python run_inference_bert_real.py

    # 5. Run the built-in demo set
    python run_inference_bert_real.py --demo

    # 6. Non-default model location
    python run_inference_bert_real.py --model-dir path/to/fine_tuned_bert_real --test "..."
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Model output layout  (mirrors finetune_bert_real.py OUTPUT_DIR)
# ---------------------------------------------------------------------------
MODEL_DIR = SCRIPT_DIR / "fine_tuned_bert_real"

_DEFAULT_ONNX_SUBDIR = "bert_real_onnx"
_DEFAULT_OV_SUBDIR   = "bert_real_openvino"

MAX_LENGTH = 128   # must match finetune_bert_real.py MAX_LENGTH


def _onnx_path(model_dir: Path) -> Path:
    """ONNX export path -- sibling of model_dir."""
    return model_dir.parent / _DEFAULT_ONNX_SUBDIR / "model.onnx"


def _ov_model_path(model_dir: Path) -> Path:
    """OpenVINO IR model.xml path."""
    return model_dir.parent / _DEFAULT_OV_SUBDIR / "model.xml"


def _ov_dir_path(model_dir: Path) -> Path:
    return model_dir.parent / _DEFAULT_OV_SUBDIR


# ---------------------------------------------------------------------------
# Label schema  (must match finetune_bert_real.py LABEL_NAMES)
# ---------------------------------------------------------------------------
LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

# Continuous score mapped to each label for weighted-average scoring
LABEL_SCORES = {
    0: -1.0,   # Negative
    1:  0.0,   # Neutral
    2:  1.0,   # Positive
}


# =============================================================================
# BACKEND DETECTION
# =============================================================================

def _detect_backends() -> dict[str, bool]:
    """Detect which inference backends are available on this machine."""
    backends = {
        "pytorch":  False,
        "onnx":     False,
        "openvino": False,
        "ov_npu":   False,
        "ov_gpu":   False,
    }

    try:
        import torch           # noqa: F401
        backends["pytorch"] = True
    except ImportError:
        pass

    try:
        import onnxruntime     # noqa: F401
        backends["onnx"] = True
    except ImportError:
        pass

    try:
        import openvino as ov
        core = ov.Core()
        available = core.available_devices
        backends["openvino"] = True
        if "NPU" in available:
            backends["ov_npu"] = True
            log.info("NPU detected: %s", core.get_property("NPU", "FULL_DEVICE_NAME"))
        if "GPU" in available:
            backends["ov_gpu"] = True
    except Exception:
        pass

    return backends


# =============================================================================
# EXPORT: PyTorch -> ONNX -> OpenVINO
# =============================================================================

def export_to_onnx(model_dir: Path,
                   output_path: Path | None = None,
                   max_length: int = MAX_LENGTH) -> bool:
    """Export the fine-tuned BERT model to ONNX format."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        log.error("PyTorch + Transformers are required for ONNX export.")
        return False

    if output_path is None:
        output_path = _onnx_path(model_dir)

    model_dir_str = str(model_dir.resolve())
    log.info("Loading model from %s", model_dir_str)

    tokenizer = AutoTokenizer.from_pretrained(model_dir_str, use_fast=True)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir_str)
    model.eval()

    # Verify label count matches training config
    num_labels = model.config.num_labels
    if num_labels != len(LABEL_NAMES):
        log.warning(
            "Model has %d labels but script expects %d -- check LABEL_NAMES",
            num_labels, len(LABEL_NAMES),
        )

    # Dummy input for tracing
    dummy_text = "The lecture was clear and well-paced overall."
    inputs = tokenizer(
        dummy_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Exporting to ONNX: %s", output_path)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits":         {0: "batch"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("ONNX export complete: %.1f MB -> %s", size_mb, output_path)
    return True


def export_to_openvino(onnx_path: Path,
                       output_dir: Path,
                       quantize: bool = True) -> bool:
    """Convert ONNX model to OpenVINO IR, optionally with INT8 quantization."""
    try:
        import openvino as ov
    except ImportError:
        log.warning("OpenVINO not installed -- skipping OV export.")
        log.warning("Install: pip install openvino --break-system-packages")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    model_xml = output_dir / "model.xml"

    log.info("Converting ONNX -> OpenVINO IR: %s", model_xml)
    try:
        core  = ov.Core()
        model = core.read_model(str(onnx_path))

        if quantize:
            log.info("Applying INT8 post-training quantization ...")
            try:
                import nncf
                quantized = nncf.quantize(
                    model,
                    calibration_dataset=_build_calibration_dataset(),
                    model_type=nncf.ModelType.TRANSFORMER,
                    preset=nncf.QuantizationPreset.MIXED,
                )
                ov.save_model(quantized, str(model_xml))
                log.info("INT8 quantization applied (NNCF)")
            except ImportError:
                log.warning("NNCF not available -- saving FP32 OpenVINO model.")
                log.warning("Install: pip install nncf --break-system-packages")
                ov.save_model(model, str(model_xml))
            except Exception as exc:
                log.warning("Quantization failed (%s) -- saving FP32 model.", exc)
                ov.save_model(model, str(model_xml))
        else:
            ov.save_model(model, str(model_xml))

        size_mb = sum(
            p.stat().st_size for p in output_dir.iterdir()
        ) / (1024 * 1024)
        log.info("OpenVINO export complete: %.1f MB total in %s", size_mb, output_dir)
        return True

    except Exception as exc:
        log.error("OpenVINO conversion failed: %s", exc)
        return False


def _build_calibration_dataset():
    """Small calibration set for PTQ quantization (covers all 3 classes)."""
    return [
        "The lecture was completely confusing and hard to follow.",
        "I did not understand anything the professor said today.",
        "The class was average -- nothing special but acceptable.",
        "It was okay overall, not great but not terrible either.",
        "Brilliant session, very well explained and engaging.",
        "I really enjoyed this class, everything made sense.",
        "Terrible lecture, worst I have ever attended.",
        "Pretty good explanation of the concepts.",
        "Neutral feelings about today's session.",
        "Absolutely loved the interactive approach.",
    ]


# =============================================================================
# INFERENCE BACKENDS
# =============================================================================

class _PyTorchBackend:
    """Raw PyTorch inference backend."""

    def __init__(self, model_dir: Path, max_length: int = MAX_LENGTH):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir_str = str(model_dir.resolve())
        self.tokenizer  = AutoTokenizer.from_pretrained(model_dir_str, use_fast=True)
        self.model      = AutoModelForSequenceClassification.from_pretrained(model_dir_str)
        self.model.eval()
        self.max_length = max_length
        self.device     = torch.device("cpu")

        n = os.cpu_count() or 4
        import torch as _t
        _t.set_num_threads(n)
        log.info("PyTorch backend loaded (%d threads)", n)

    def predict_batch(self, texts: list[str]) -> np.ndarray:
        import torch
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**enc).logits
        return logits.numpy()

    @property
    def name(self) -> str:
        return "pytorch"


class _ONNXBackend:
    """ONNX Runtime inference backend."""

    def __init__(self, onnx_path: Path,
                 tokenizer_dir: Path,
                 max_length: int = MAX_LENGTH):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads      = os.cpu_count() or 4
        opts.inter_op_num_threads      = max(1, (os.cpu_count() or 4) // 2)
        opts.execution_mode            = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level  = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session    = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer  = AutoTokenizer.from_pretrained(
            str(tokenizer_dir.resolve()), use_fast=True
        )
        self.max_length = max_length
        log.info("ONNX Runtime backend loaded")

    def predict_batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        inputs = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = self.session.run(["logits"], inputs)[0]
        return logits

    @property
    def name(self) -> str:
        return "onnx"


class _OpenVINOBackend:
    """OpenVINO inference backend -- CPU, NPU, or GPU."""

    def __init__(self, model_xml: Path,
                 tokenizer_dir: Path,
                 device: str = "CPU",
                 max_length: int = MAX_LENGTH):
        import openvino as ov
        from transformers import AutoTokenizer

        core  = ov.Core()
        model = core.read_model(str(model_xml))

        config = {"PERFORMANCE_HINT": "THROUGHPUT"}
        if device == "CPU":
            config["CPU_THROUGHPUT_STREAMS"] = "AUTO"
            config["INFERENCE_NUM_THREADS"]  = str(os.cpu_count() or 4)
        elif device == "NPU":
            config = {"PERFORMANCE_HINT": "LATENCY"}

        self.compiled   = core.compile_model(model, device, config)
        self.tokenizer  = AutoTokenizer.from_pretrained(
            str(tokenizer_dir.resolve()), use_fast=True
        )
        self.max_length = max_length
        self.device_str = device
        log.info("OpenVINO backend loaded (device=%s)", device)

    def predict_batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        inputs = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        result = self.compiled(inputs)
        return list(result.values())[0]

    @property
    def name(self) -> str:
        return f"openvino_{self.device_str.lower()}"


# =============================================================================
# BACKEND AUTO-SELECTOR
# =============================================================================

def _load_best_backend(model_dir: Path,
                       preferred: str | None = None) -> Any:
    """
    Auto-selects the fastest available backend.
    Priority: OpenVINO NPU > OpenVINO CPU > ONNX Runtime > PyTorch
    """
    backends      = _detect_backends()
    tokenizer_dir = model_dir
    ov_model      = _ov_model_path(model_dir)
    onnx_path     = _onnx_path(model_dir)

    # NPU -- fastest on supported Intel hardware
    if (preferred in (None, "npu", "openvino")
            and backends["ov_npu"] and ov_model.exists()):
        try:
            return _OpenVINOBackend(ov_model, tokenizer_dir, device="NPU")
        except Exception as exc:
            log.warning("NPU backend failed: %s -- falling back", exc)

    # OpenVINO CPU -- typically 2-4x faster than raw PyTorch on Intel CPUs
    if (preferred in (None, "openvino", "cpu_ov")
            and backends["openvino"] and ov_model.exists()):
        try:
            return _OpenVINOBackend(ov_model, tokenizer_dir, device="CPU")
        except Exception as exc:
            log.warning("OpenVINO CPU backend failed: %s -- falling back", exc)

    # ONNX Runtime
    if (preferred in (None, "onnx")
            and backends["onnx"] and onnx_path.exists()):
        try:
            return _ONNXBackend(onnx_path, tokenizer_dir)
        except Exception as exc:
            log.warning("ONNX backend failed: %s -- falling back", exc)

    # Raw PyTorch -- config.json is saved by finetune_bert_real.py
    config_file = model_dir / "config.json"
    if backends["pytorch"] and config_file.exists():
        try:
            return _PyTorchBackend(model_dir)
        except Exception as exc:
            log.warning("PyTorch backend failed: %s", exc)

    raise RuntimeError(
        f"No inference backend is available.\n"
        f"  model_dir   : {model_dir}\n"
        f"  config.json : "
        f"{'FOUND' if (model_dir / 'config.json').exists() else 'MISSING -- run finetune_bert_real.py first'}\n"
        f"  onnx model  : "
        f"{'found' if onnx_path.exists() else 'not found -- run --export after training'}\n"
        f"  ov model    : "
        f"{'found' if ov_model.exists() else 'not found -- run --export after training'}\n"
        f"  pytorch     : "
        f"{'available' if backends['pytorch'] else 'not installed -- pip install torch'}"
    )


# =============================================================================
# SOFTMAX / SCORING UTILS
# =============================================================================

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _logits_to_prediction(logits: np.ndarray) -> dict:
    """
    Convert a 1-D logits array (length = num_labels) to a prediction dict.

    Returns:
        label_id    : int in {0, 1, 2}
        label_name  : str  "Negative" | "Neutral" | "Positive"
        confidence  : float in [0, 1]  -- probability of predicted class
        score       : float in [-1, +1] -- weighted average across all classes
        all_probs   : dict {label_name: probability}
    """
    probs      = _softmax(logits)
    label_id   = int(np.argmax(probs))
    confidence = float(probs[label_id])

    # Continuous score: weighted average of per-label anchor scores
    anchors = np.array([LABEL_SCORES[i] for i in range(len(LABEL_SCORES))])
    score   = float(np.dot(probs, anchors))

    return {
        "label_id":   label_id,
        "label_name": LABEL_NAMES[label_id],
        "confidence": round(confidence, 4),
        "score":      round(score, 4),
        "all_probs":  {
            LABEL_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(LABEL_NAMES))
        },
    }


# =============================================================================
# MAIN INFERENCE ENGINE
# =============================================================================

class SentimentInferenceEngine:
    """
    3-class sentiment inference engine for the BERT model trained by
    finetune_bert_real.py.

    Public API:
        engine.predict(text)            -> single prediction dict
        engine.predict_batch(texts)     -> list of prediction dicts
        engine.predict_proba(text)      -> raw probabilities dict
        engine.warmup()                 -> pre-warm backend (reduces first-call latency)
    """

    def __init__(self,
                 model_dir: str | Path = MODEL_DIR,
                 backend:   str | None = None,
                 batch_size: int = 16):
        """
        Args:
            model_dir:  Path to fine_tuned_bert_real/ (or any compatible save dir).
            backend:    Force a backend: "pytorch", "onnx", "openvino", "npu".
                        None = auto-select fastest available.
            batch_size: Batch size for predict_batch().
        """
        self.model_dir  = Path(model_dir)
        self.batch_size = batch_size
        self._backend   = _load_best_backend(self.model_dir, backend)
        log.info("Active backend: %s", self._backend.name)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a single text string.

        Returns dict with keys:
            text, label_id, label_name, confidence, score, all_probs
        """
        if not text or not text.strip():
            return {
                "text": text,
                "label_id": 1,
                "label_name": "Neutral",
                "confidence": 0.0,
                "score": 0.0,
                "all_probs": {n: 0.0 for n in LABEL_NAMES.values()},
                "error": "Empty input",
            }

        logits = self._backend.predict_batch([text.strip()])
        result = _logits_to_prediction(logits[0])
        result["text"] = text
        return result

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Predict sentiment for a list of texts. Runs in mini-batches.

        Returns a list of prediction dicts (same structure as predict()).
        """
        if not texts:
            return []

        all_results = []
        for start in range(0, len(texts), self.batch_size):
            batch  = texts[start : start + self.batch_size]
            logits = self._backend.predict_batch(batch)
            for i, text in enumerate(batch):
                result       = _logits_to_prediction(logits[i])
                result["text"] = text
                all_results.append(result)

        return all_results

    def predict_proba(self, text: str) -> dict[str, float]:
        """
        Return softmax probabilities for all 3 classes.

        Returns:
            {"Negative": p0, "Neutral": p1, "Positive": p2}
        """
        result = self.predict(text)
        return result.get("all_probs", {})

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self, n: int = 3):
        """
        Run n dummy inferences to eliminate first-call JIT / compilation latency.
        Call this once after constructing the engine before timed inference.
        """
        dummy = "The class was fairly typical with some good and some bad moments."
        for _ in range(n):
            self._backend.predict_batch([dummy])
        log.info("Backend warmed up (%d passes)", n)

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else "none"


# =============================================================================
# BENCHMARKING
# =============================================================================

def benchmark_backends(model_dir: Path, n_runs: int = 50):
    """
    Benchmark all available inference backends.
    Prints mean latency (ms), std (ms), and throughput (samples/sec).
    """
    backends = _detect_backends()

    # Representative sentences covering all 3 classes
    test_texts = [
        "The lecture was completely incomprehensible and I gave up halfway through.",
        "Okay class, nothing exceptional but it covered the basics adequately.",
        "Absolutely brilliant session -- crystal clear and very engaging.",
        "I am totally lost and have no idea what is going on in this course.",
        "Average experience, some parts were good others less so.",
        "Loved the interactive approach, best class this semester by far.",
        "The slides were way too dense and the explanation was confusing.",
        "It was fine I suppose, not great but not terrible either.",
    ] * 2  # batch of 16

    results = {}
    print("\n" + "=" * 62)
    print("  BACKEND BENCHMARK")
    print(f"  {n_runs} runs x batch_size={len(test_texts)}")
    print("=" * 62)

    def _run(backend, name: str):
        # Warmup
        for _ in range(3):
            backend.predict_batch(test_texts[:2])
        # Timed runs
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            backend.predict_batch(test_texts)
            latencies.append((time.perf_counter() - t0) * 1000)
        mean_ms = float(np.mean(latencies))
        std_ms  = float(np.std(latencies))
        tput    = (len(test_texts) * 1000) / mean_ms
        results[name] = {
            "mean_ms":        round(mean_ms, 2),
            "std_ms":         round(std_ms,  2),
            "samples_per_sec": round(tput, 1),
        }
        print(f"  {name:<25}  {mean_ms:7.2f} +/- {std_ms:5.2f} ms   "
              f"{tput:7.1f} samples/sec")

    tokenizer_dir = model_dir
    ov_model      = _ov_model_path(model_dir)
    onnx_path     = _onnx_path(model_dir)

    if ov_model.exists() and backends["ov_npu"]:
        try:
            b = _OpenVINOBackend(ov_model, tokenizer_dir, "NPU")
            _run(b, "openvino_npu")
        except Exception as exc:
            print(f"  openvino_npu              FAILED: {exc}")

    if ov_model.exists() and backends["openvino"]:
        try:
            b = _OpenVINOBackend(ov_model, tokenizer_dir, "CPU")
            _run(b, "openvino_cpu")
        except Exception as exc:
            print(f"  openvino_cpu              FAILED: {exc}")

    if onnx_path.exists() and backends["onnx"]:
        try:
            b = _ONNXBackend(onnx_path, tokenizer_dir)
            _run(b, "onnxruntime")
        except Exception as exc:
            print(f"  onnxruntime               FAILED: {exc}")

    if (model_dir / "config.json").exists() and backends["pytorch"]:
        try:
            b = _PyTorchBackend(model_dir)
            _run(b, "pytorch_cpu")
        except Exception as exc:
            print(f"  pytorch_cpu               FAILED: {exc}")

    if results:
        fastest = min(results, key=lambda k: results[k]["mean_ms"])
        print(f"\n  Recommended backend: {fastest} "
              f"({results[fastest]['mean_ms']:.1f} ms per batch)")
    print("=" * 62 + "\n")
    return results


# =============================================================================
# CLI DISPLAY HELPER
# =============================================================================

def _print_result(result: dict, show_probs: bool = False):
    """Pretty-print a single prediction result."""
    _win = platform.system() == "Windows"
    pol_sym = {
        "Negative": "[-]",
        "Neutral":  "[ ]",
        "Positive": "[+]",
    } if _win else {
        "Negative": "X",
        "Neutral":  "O",
        "Positive": "+",
    }

    text       = result.get("text", "")[:80]
    label      = result.get("label_name", "?")
    score      = result.get("score", 0.0)
    confidence = result.get("confidence", 0.0)
    sym        = pol_sym.get(label, "?")

    # Score bar: [-1, +1] mapped to 20 chars
    n_blocks  = int((score + 1.0) * 10)
    n_blocks  = max(0, min(20, n_blocks))
    bar       = ("#" * n_blocks + "." * (20 - n_blocks)) if _win else ("*" * n_blocks)

    print(f'\n  Input      : "{text}"')
    print(f"  Prediction : {sym} {label:<10}  score={score:+.4f}  conf={confidence:.4f}  [{bar}]")

    if show_probs:
        probs = result.get("all_probs", {})
        for lname, prob in probs.items():
            bar_p   = int(prob * 20)
            bar_str = "#" * bar_p
            print(f"    {lname:<12} {prob:.4f}  {bar_str}")


# =============================================================================
# SINGLETON / INTEGRATION HELPER
# =============================================================================

_GLOBAL_ENGINE: SentimentInferenceEngine | None = None


def get_engine(model_dir: str | Path = MODEL_DIR,
               backend: str | None = None) -> SentimentInferenceEngine:
    """
    Get or create the global singleton inference engine.
    Use this for long-running applications to avoid repeated model loads.
    """
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = SentimentInferenceEngine(model_dir, backend=backend)
        _GLOBAL_ENGINE.warmup()
    return _GLOBAL_ENGINE


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BERT 3-class sentiment inference -- compatible with finetune_bert_real.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference_bert_real.py --export
  python run_inference_bert_real.py --benchmark
  python run_inference_bert_real.py --demo
  python run_inference_bert_real.py --test "The lecture was incredibly clear"
  python run_inference_bert_real.py --model-dir path/to/fine_tuned_bert_real --test "..."
  python run_inference_bert_real.py --backend pytorch --test "Average session today"
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--export", action="store_true",
        help="Export PyTorch model -> ONNX -> OpenVINO (run once after training)",
    )
    group.add_argument(
        "--benchmark", action="store_true",
        help="Benchmark all available backends and print recommendation",
    )
    group.add_argument(
        "--test", type=str, default=None,
        help="Score a single text and exit",
    )
    group.add_argument(
        "--demo", action="store_true",
        help="Run the built-in demo set of representative sentences",
    )

    parser.add_argument(
        "--model-dir", type=str, default=str(MODEL_DIR),
        help="Path to fine_tuned_bert_real/ directory (default: ./fine_tuned_bert_real)",
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        choices=["pytorch", "onnx", "openvino", "npu"],
        help="Force a specific inference backend (default: auto)",
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Skip INT8 quantization during --export",
    )
    parser.add_argument(
        "--max-length", type=int, default=MAX_LENGTH,
        help=f"Max token length for ONNX export (default: {MAX_LENGTH})",
    )
    parser.add_argument(
        "--probs", action="store_true",
        help="Show all class probabilities in addition to the top prediction",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Mini-batch size for inference (default: 16)",
    )

    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    if args.export:
        print("=" * 60)
        print("  EXPORT PIPELINE: PyTorch -> ONNX -> OpenVINO")
        print("=" * 60)
        onnx_out = _onnx_path(model_dir)
        ov_out   = _ov_dir_path(model_dir)
        ok = export_to_onnx(model_dir, onnx_out, args.max_length)
        if ok:
            export_to_openvino(onnx_out, ov_out, quantize=not args.no_quantize)
        return

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    if args.benchmark:
        benchmark_backends(model_dir)
        return

    # ------------------------------------------------------------------
    # Build engine for test / demo / interactive
    # ------------------------------------------------------------------
    engine = SentimentInferenceEngine(
        model_dir  = model_dir,
        backend    = args.backend,
        batch_size = args.batch_size,
    )
    engine.warmup()

    # ------------------------------------------------------------------
    # Single test
    # ------------------------------------------------------------------
    if args.test:
        t0 = time.perf_counter()
        result = engine.predict(args.test)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _print_result(result, show_probs=args.probs)
        print(f"\n  Latency: {elapsed_ms:.1f} ms  (backend: {engine.backend_name})")
        return

    # ------------------------------------------------------------------
    # Demo
    # ------------------------------------------------------------------
    if args.demo:
        demo_texts = [
            # Clearly negative
            "I have no idea what is happening in this class. Completely lost.",
            "The explanation was terrible and I understood nothing at all.",
            "Worst lecture I have attended this entire semester, very confusing.",
            # Mixed / borderline
            "The lecture was okay, not amazing but I managed to follow along.",
            "Some parts were good but others were a bit unclear to me.",
            "It was a standard session, nothing special but covered the material.",
            # Clearly positive
            "Absolutely brilliant class, everything was clear and well explained.",
            "I really enjoyed today's session, the best of the semester so far.",
            "Great lecture, very engaging and I left feeling confident.",
            # Edge cases
            "ok",
            "fine I guess",
            "The class happened and I was present.",
            # Sarcasm (hard for any model)
            "Oh fantastic, another lecture where I understood nothing. Really helpful.",
            "Amazing how little I learned in 90 minutes. Incredible efficiency.",
            # Multi-sentence
            (
                "The first half was excellent and very clear. "
                "The second half completely lost me and I could not follow."
            ),
        ]

        print("\n" + "=" * 62)
        print("  DEMO -- BERT 3-class Sentiment Engine")
        print(f"  Backend: {engine.backend_name}  |  Labels: Negative / Neutral / Positive")
        print("=" * 62)

        total_t = 0.0
        for text in demo_texts:
            t0 = time.perf_counter()
            result = engine.predict(text)
            ms = (time.perf_counter() - t0) * 1000
            total_t += ms
            _print_result(result, show_probs=args.probs)
            print(f"  [{ms:.1f} ms]")

        print(f"\n  Total  : {total_t:.1f} ms for {len(demo_texts)} samples")
        print(f"  Average: {total_t / len(demo_texts):.1f} ms/sample")
        print("=" * 62)
        return

    # ------------------------------------------------------------------
    # Interactive mode (default)
    # ------------------------------------------------------------------
    print(f"\n  BERT Sentiment Engine -- Interactive Mode")
    print(f"  Backend : {engine.backend_name}")
    print(f"  Labels  : Negative (0) | Neutral (1) | Positive (2)")
    print(f"  Model   : {model_dir}")
    print("  Type text and press Enter. 'quit' to exit.\n")

    while True:
        try:
            text = input("  Text > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break

        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        t0 = time.perf_counter()
        result = engine.predict(text)
        ms = (time.perf_counter() - t0) * 1000
        _print_result(result, show_probs=args.probs)
        print(f"  [{ms:.1f} ms -- {engine.backend_name}]\n")


if __name__ == "__main__":
    main()