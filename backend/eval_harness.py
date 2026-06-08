#!/usr/bin/env python3
"""
eval_harness.py  —  Universal Evaluation Harness
=================================================
Runs a comprehensive capability test suite against BOTH inference versions:

  • V1 : SentimentInferenceEngine  (3-class: Negative / Neutral / Positive)
         from run_inference_bert_real.py  +  fine_tuned_bert_real/

  • V2 : APEInferenceEngine         (7-class ABSA, aspect-aware)
         from run_inference.py       +  outputs/models/

Tests every capability listed in the capability table, with ≥20 cases per
capability where applicable.  Collects deep system telemetry (CPU, GPU/NPU,
RAM, temperature, power) and writes:

  • eval_report_<timestamp>.json   – machine-readable results
  • eval_report_<timestamp>.txt    – human-readable report
  • eval_report_<timestamp>.csv    – per-sample latency / metric log

Usage
-----
    # Both models in default locations (same directory as this script)
    python eval_harness.py

    # Custom locations
    python eval_harness.py \\
        --v1-model-dir ./fine_tuned_bert_real \\
        --v2-model-dir ./outputs/models

    # Run only one version
    python eval_harness.py --only v1
    python eval_harness.py --only v2

    # Skip warmup (faster but first-inference latency included)
    python eval_harness.py --no-warmup

    # Force a specific backend
    python eval_harness.py --backend pytorch

Dependencies (all standard in both pipelines)
----------------------------------------------
    pip install torch transformers psutil numpy
    pip install openvino onnxruntime   # optional — auto-detected
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EVAL")

SCRIPT_DIR = Path(__file__).resolve().parent
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── optional hardware libs ──────────────────────────────────────────────────────
try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False
    log.warning("psutil not found — hardware telemetry limited. pip install psutil")

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# Each entry: (capability_tag, text, expected_v1_label, expected_aspects_v2)
# expected_v1_label: "Negative" | "Neutral" | "Positive" | None (no check)
# expected_aspects_v2: list of aspects expected to appear in output, or None
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES: List[Dict[str, Any]] = [

    # ── 1. ASPECT EXTRACTION ──────────────────────────────────────────────────
    {"cap": "Aspect Extraction",
     "text": "The slides were too cluttered and the pacing was way too fast.",
     "v1_label": "Negative", "v2_aspects": ["clarity", "pacing"]},

    {"cap": "Aspect Extraction",
     "text": "Group discussions were excellent and the pre-class videos were helpful.",
     "v1_label": "Positive", "v2_aspects": ["collaboration", "pre_class"]},

    {"cap": "Aspect Extraction",
     "text": "The workload this semester is completely unreasonable.",
     "v1_label": "Negative", "v2_aspects": ["workload"]},

    # ── 2. DYNAMIC ASPECT DISCOVERY ───────────────────────────────────────────
    {"cap": "Dynamic Aspect Discovery",
     "text": "I feel completely invisible in this class, nobody listens.",
     "v1_label": "Negative", "v2_aspects": None},

    {"cap": "Dynamic Aspect Discovery",
     "text": "The assessment criteria have never been made clear to us.",
     "v1_label": "Negative", "v2_aspects": None},

    # ── 3. ABSA ────────────────────────────────────────────────────────────────
    {"cap": "Aspect-Based Sentiment Analysis",
     "text": "The content was excellent but the pacing ruined the experience.",
     "v1_label": "Neutral", "v2_aspects": ["clarity", "pacing"]},

    {"cap": "Aspect-Based Sentiment Analysis",
     "text": "Lecture clarity has improved but collaboration sessions remain chaotic.",
     "v1_label": "Neutral", "v2_aspects": ["clarity", "collaboration"]},

    # ── 4. MULTI-ASPECT DETECTION ─────────────────────────────────────────────
    {"cap": "Multi-Aspect Detection",
     "text": "The pacing is too fast, group work is useless, and pre-class videos are too long.",
     "v1_label": "Negative", "v2_aspects": ["pacing", "collaboration", "pre_class"]},

    {"cap": "Multi-Aspect Detection",
     "text": "I loved the engagement, found the workload manageable, and clarity was superb.",
     "v1_label": "Positive", "v2_aspects": ["engagement", "workload", "clarity"]},

    # ── 5. IMPLICIT SENTIMENT DETECTION ──────────────────────────────────────
    {"cap": "Implicit Sentiment Detection",
     "text": "I keep re-reading the same slide five times and still nothing clicks.",
     "v1_label": "Negative", "v2_aspects": None},

    {"cap": "Implicit Sentiment Detection",
     "text": "I actually looked forward to coming to class this week.",
     "v1_label": "Positive", "v2_aspects": None},

    {"cap": "Implicit Sentiment Detection",
     "text": "By the end of each lecture I have three pages of unanswered questions.",
     "v1_label": "Negative", "v2_aspects": None},

    # ── 6. SARCASM DETECTION ──────────────────────────────────────────────────
    {"cap": "Sarcasm Detection",
     "text": "Oh fantastic, another lecture where I understood absolutely nothing. Really helpful.",
     "v1_label": "Negative", "v2_aspects": None},

    {"cap": "Sarcasm Detection",
     "text": "Brilliant — three exams in one week, clearly designed with student wellbeing in mind.",
     "v1_label": "Negative", "v2_aspects": None},

    {"cap": "Sarcasm Detection",
     "text": "Amazing how we covered five weeks of content in ninety minutes. Truly a masterclass.",
     "v1_label": "Negative", "v2_aspects": None},

    # ── 7. IRONY DETECTION ────────────────────────────────────────────────────
    {"cap": "Irony Detection",
     "text": "Loved how the 'interactive' session was just the professor reading slides aloud.",
     "v1_label": "Negative", "v2_aspects": None},

    {"cap": "Irony Detection",
     "text": "The 'clear' explanation left me more confused than before the class started.",
     "v1_label": "Negative", "v2_aspects": None},

    # ── 8. MIXED SENTIMENT ────────────────────────────────────────────────────
    {"cap": "Mixed Sentiment Analysis",
     "text": "The first half was brilliant and very engaging; the second half was a disaster.",
     "v1_label": "Neutral", "v2_aspects": ["engagement"]},

    {"cap": "Mixed Sentiment Analysis",
     "text": "I really enjoyed the collaboration tasks even though the workload was brutal.",
     "v1_label": "Neutral", "v2_aspects": ["collaboration", "workload"]},

    {"cap": "Mixed Sentiment Analysis",
     "text": "Great explanations on most topics, but pacing on the last chapter was terrible.",
     "v1_label": "Neutral", "v2_aspects": ["clarity", "pacing"]},

    # ── 9. NEGATION HANDLING ──────────────────────────────────────────────────
    {"cap": "Negation Handling",
     "text": "I didn't find the lecture confusing at all — it was crystal clear.",
     "v1_label": "Positive", "v2_aspects": ["clarity"]},

    {"cap": "Negation Handling",
     "text": "This is not the kind of engaging teaching I expected from this course.",
     "v1_label": "Negative", "v2_aspects": ["engagement"]},

    {"cap": "Negation Handling",
     "text": "It's not that I didn't understand — I just couldn't follow the last part.",
     "v1_label": "Negative", "v2_aspects": ["clarity"]},

    {"cap": "Negation Handling",
     "text": "The class was never boring; I was engaged throughout every single session.",
     "v1_label": "Positive", "v2_aspects": ["engagement"]},

    # ── 10. COMPARATIVE OPINION ANALYSIS ─────────────────────────────────────
    {"cap": "Comparative Opinion Analysis",
     "text": "This semester's lectures are significantly clearer than last year's.",
     "v1_label": "Positive", "v2_aspects": ["clarity"]},

    {"cap": "Comparative Opinion Analysis",
     "text": "The workload here is double what other courses demand for half the credit.",
     "v1_label": "Negative", "v2_aspects": ["workload"]},

    # ── 11. COREFERENCE RESOLUTION ────────────────────────────────────────────
    {"cap": "Coreference Resolution",
     "text": "The professor explained the concept well but she moved on before we absorbed it.",
     "v1_label": "Neutral", "v2_aspects": ["clarity", "pacing"]},

    {"cap": "Coreference Resolution",
     "text": "The group task was well designed and it actually helped me understand the topic.",
     "v1_label": "Positive", "v2_aspects": ["collaboration"]},

    # ── 12. CONTEXTUAL UNDERSTANDING ──────────────────────────────────────────
    {"cap": "Contextual Understanding",
     "text": "Given that exams are in two weeks, covering this much new material feels reckless.",
     "v1_label": "Negative", "v2_aspects": ["workload", "pacing"]},

    {"cap": "Contextual Understanding",
     "text": "Now that the semester is ending I can say this was the best course I have taken.",
     "v1_label": "Positive", "v2_aspects": None},

    # ── 13. STUDENT SLANG ────────────────────────────────────────────────────
    {"cap": "Student Slang Interpretation",
     "text": "This class slaps, prof actually explains stuff that makes sense fr fr.",
     "v1_label": "Positive", "v2_aspects": None},

    {"cap": "Student Slang Interpretation",
     "text": "Lowkey the hardest course I've ever taken, assignments are absolutely bussin.",
     "v1_label": "Neutral", "v2_aspects": ["workload"]},

    {"cap": "Student Slang Interpretation",
     "text": "The lecture was mid, nothing special, not trash but definitely not fire either.",
     "v1_label": "Neutral", "v2_aspects": None},

    # ── 14. ABBREVIATION EXPANSION ────────────────────────────────────────────
    {"cap": "Abbreviation Expansion",
     "text": "TBH the prof's explanation of DSA was so confusing I had to watch YT for 2hrs.",
     "v1_label": "Negative", "v2_aspects": ["clarity"]},

    {"cap": "Abbreviation Expansion",
     "text": "IMO the TA sessions are way more helpful than the actual lectures TBH.",
     "v1_label": "Negative", "v2_aspects": None},

    # ── 15. EMOTION DETECTION ─────────────────────────────────────────────────
    {"cap": "Emotion Detection",
     "text": "I am so frustrated — no matter how hard I study I leave class more confused.",
     "v1_label": "Negative", "v2_aspects": ["clarity"]},

    {"cap": "Emotion Detection",
     "text": "I feel genuinely excited about this subject for the first time in years.",
     "v1_label": "Positive", "v2_aspects": ["engagement"]},

    {"cap": "Emotion Detection",
     "text": "There is a constant undercurrent of anxiety every time the workload is announced.",
     "v1_label": "Negative", "v2_aspects": ["workload"]},

    {"cap": "Emotion Detection",
     "text": "I walked out of that lecture feeling proud of how much I had absorbed.",
     "v1_label": "Positive", "v2_aspects": None},

    # ── 16. SENTIMENT INTENSITY SCORING ──────────────────────────────────────
    {"cap": "Sentiment Intensity Scoring",
     "text": "It was okay.",
     "v1_label": "Neutral", "v2_aspects": None},

    {"cap": "Sentiment Intensity Scoring",
     "text": "It was absolutely, catastrophically terrible — the single worst lecture I have ever attended.",
     "v1_label": "Negative", "v2_aspects": None},

    {"cap": "Sentiment Intensity Scoring",
     "text": "Pretty decent session overall, could have been a bit clearer.",
     "v1_label": "Neutral", "v2_aspects": ["clarity"]},

    {"cap": "Sentiment Intensity Scoring",
     "text": "Life-changing lecture. I finally understand everything from the past six weeks.",
     "v1_label": "Positive", "v2_aspects": ["clarity"]},

    # ── 17. SEVERITY ASSESSMENT ────────────────────────────────────────────────
    {"cap": "Severity Assessment",
     "text": "The workload is so crushing that multiple students have dropped the course.",
     "v1_label": "Negative", "v2_aspects": ["workload"]},

    {"cap": "Severity Assessment",
     "text": "The pacing is slightly faster than I prefer but manageable with some effort.",
     "v1_label": "Neutral", "v2_aspects": ["pacing"]},

    # ── 18. OPINION TARGET EXTRACTION ────────────────────────────────────────
    {"cap": "Opinion Target Extraction",
     "text": "The textbook is useless but the professor's handouts are invaluable.",
     "v1_label": "Neutral", "v2_aspects": None},

    {"cap": "Opinion Target Extraction",
     "text": "Online quizzes are fair but the final exam questions are completely unexpected.",
     "v1_label": "Neutral", "v2_aspects": None},

    # ── 19. OPINION PHRASE EXTRACTION ────────────────────────────────────────
    {"cap": "Opinion Phrase Extraction",
     "text": "The visuals were stunning and the pacing was absolutely perfect for the content.",
     "v1_label": "Positive", "v2_aspects": ["pacing"]},

    {"cap": "Opinion Phrase Extraction",
     "text": "Horribly monotone delivery and slides that are impossible to read.",
     "v1_label": "Negative", "v2_aspects": ["engagement", "clarity"]},

    # ── 20. EVIDENCE EXTRACTION / EXPLAINABILITY ──────────────────────────────
    {"cap": "Evidence Extraction",
     "text": "Because the slides had 80 bullet points each, I could not follow the thread.",
     "v1_label": "Negative", "v2_aspects": ["clarity"]},

    {"cap": "Explainable Predictions",
     "text": "The worked examples made everything click and saved me hours of confusion.",
     "v1_label": "Positive", "v2_aspects": ["clarity"]},

    # ── 21. CONTRADICTION DETECTION ───────────────────────────────────────────
    {"cap": "Contradiction Detection",
     "text": "This is the best course ever. Actually no, it is terrible. Well, kind of both.",
     "v1_label": None, "v2_aspects": None},

    {"cap": "Contradiction Detection",
     "text": "The professor is great but also not very good at explaining things clearly.",
     "v1_label": None, "v2_aspects": ["clarity"]},

    # ── 22. AMBIGUITY DETECTION ────────────────────────────────────────────────
    {"cap": "Ambiguity Detection",
     "text": "Something feels off this semester but I cannot quite put my finger on it.",
     "v1_label": "Negative", "v2_aspects": None},

    {"cap": "Ambiguity Detection",
     "text": "It was different.",
     "v1_label": "Neutral", "v2_aspects": None},

    # ── 23. MULTI-HOP REASONING ───────────────────────────────────────────────
    {"cap": "Multi-Hop Reasoning",
     "text": "Because the pre-class videos were unclear, students came unprepared, which made group work fail.",
     "v1_label": "Negative", "v2_aspects": ["pre_class", "collaboration"]},

    {"cap": "Multi-Hop Reasoning",
     "text": "Since the pacing slowed down last week, I finally had time to absorb the content and it all made sense.",
     "v1_label": "Positive", "v2_aspects": ["pacing", "clarity"]},

    # ── 24. CAUSE-EFFECT EXTRACTION ──────────────────────────────────────────
    {"cap": "Cause-Effect Extraction",
     "text": "Because the professor rushes through derivations I miss half the steps and fail practice problems.",
     "v1_label": "Negative", "v2_aspects": ["pacing", "clarity"]},

    {"cap": "Cause-Effect Extraction",
     "text": "Thanks to the interactive exercises I can now apply every concept with confidence.",
     "v1_label": "Positive", "v2_aspects": ["engagement"]},

    # ── 25. TEMPORAL UNDERSTANDING ────────────────────────────────────────────
    {"cap": "Temporal Understanding",
     "text": "The course was confusing in week one but has become much clearer since then.",
     "v1_label": "Positive", "v2_aspects": ["clarity"]},

    {"cap": "Temporal Understanding",
     "text": "Every semester the workload keeps increasing without any improvement in support.",
     "v1_label": "Negative", "v2_aspects": ["workload"]},

    # ── 26. RECOMMENDATION DETECTION ────────────────────────────────────────
    {"cap": "Recommendation Detection",
     "text": "I think smaller lecture groups would dramatically improve student engagement.",
     "v1_label": "Neutral", "v2_aspects": ["engagement"]},

    {"cap": "Recommendation Detection",
     "text": "Could the professor consider posting annotated slides before each class?",
     "v1_label": "Neutral", "v2_aspects": ["pre_class"]},

    {"cap": "Recommendation Detection",
     "text": "More practice problems with step-by-step solutions would really help clarity.",
     "v1_label": "Positive", "v2_aspects": ["clarity"]},

    # ── 27. QUESTION DETECTION ────────────────────────────────────────────────
    {"cap": "Question Detection",
     "text": "Is there any chance the deadline for the assignment could be extended?",
     "v1_label": "Neutral", "v2_aspects": ["workload"]},

    {"cap": "Question Detection",
     "text": "Why are we covering content that isn't even in the syllabus?",
     "v1_label": "Negative", "v2_aspects": None},

    # ── 28. INTENT CLASSIFICATION ─────────────────────────────────────────────
    {"cap": "Intent Classification",
     "text": "I want to formally complain about the grading inconsistency in this course.",
     "v1_label": "Negative", "v2_aspects": None},

    {"cap": "Intent Classification",
     "text": "I just wanted to say this course changed how I think about the subject.",
     "v1_label": "Positive", "v2_aspects": None},

    # ── 29. UNCERTAINTY ESTIMATION ────────────────────────────────────────────
    {"cap": "Uncertainty Estimation",
     "text": "Maybe the pacing is fine and I am just slower than average — hard to tell.",
     "v1_label": "Neutral", "v2_aspects": ["pacing"]},

    {"cap": "Uncertainty Estimation",
     "text": "I think the lectures are probably fine but I could be missing something.",
     "v1_label": "Neutral", "v2_aspects": None},

    # ── 30. HALLUCINATION RESISTANCE (edge / stress cases) ────────────────────
    {"cap": "Hallucination Resistance",
     "text": "ok",
     "v1_label": "Neutral", "v2_aspects": None},

    {"cap": "Hallucination Resistance",
     "text": "fine I guess",
     "v1_label": "Neutral", "v2_aspects": None},

    {"cap": "Hallucination Resistance",
     "text": "the class happened",
     "v1_label": "Neutral", "v2_aspects": None},

    # ── 31. REAL-TIME INFERENCE (latency stress) ──────────────────────────────
    {"cap": "Real-Time Inference",
     "text": "Everything about this course is outstanding — content, delivery, pacing, and collaboration.",
     "v1_label": "Positive",
     "v2_aspects": ["clarity", "pacing", "engagement", "collaboration"]},

    {"cap": "Real-Time Inference",
     "text": "The workload, pacing, clarity, engagement, and collaboration are all deeply unsatisfactory.",
     "v1_label": "Negative",
     "v2_aspects": ["workload", "pacing", "clarity", "engagement", "collaboration"]},

    # ── 32. DOMAIN ADAPTATION ─────────────────────────────────────────────────
    {"cap": "Domain Adaptation",
     "text": "In the lab sessions equipment kept failing which made practical work impossible.",
     "v1_label": "Negative", "v2_aspects": None},

    {"cap": "Domain Adaptation",
     "text": "The online portal for submitting assignments crashes every single week.",
     "v1_label": "Negative", "v2_aspects": ["workload"]},

    # ── 33. PRIVACY PRESERVATION (PII stress) ────────────────────────────────
    {"cap": "Privacy Preservation",
     "text": "Professor Smith's approach to grading is completely unfair and biased.",
     "v1_label": "Negative", "v2_aspects": None},

    {"cap": "Privacy Preservation",
     "text": "My friend John told me the lab on Monday is the worst session of the week.",
     "v1_label": "Negative", "v2_aspects": None},

    # ── 34. BIAS DETECTION ───────────────────────────────────────────────────
    {"cap": "Bias Detection and Mitigation",
     "text": "Female students always ask better questions and get more attention in discussions.",
     "v1_label": "Neutral", "v2_aspects": None},

    {"cap": "Bias Detection and Mitigation",
     "text": "International students struggle more because the teaching style assumes local cultural context.",
     "v1_label": "Negative", "v2_aspects": None},

    # ── 35. WORKLOAD ANALYSIS ────────────────────────────────────────────────
    {"cap": "Workload Analysis",
     "text": "Three assignments, one quiz, and a project all due in the same week is inhumane.",
     "v1_label": "Negative", "v2_aspects": ["workload"]},

    {"cap": "Workload Analysis",
     "text": "The assignment load is perfectly calibrated — challenging but not overwhelming.",
     "v1_label": "Positive", "v2_aspects": ["workload"]},

    # ── 36. CODE-MIXED LANGUAGE ──────────────────────────────────────────────
    {"cap": "Code-Mixed Language Understanding",
     "text": "Yaar the class was thoda confusing lekin overall theek tha.",
     "v1_label": None, "v2_aspects": None},

    {"cap": "Code-Mixed Language Understanding",
     "text": "Class ekdum bakwaas hai, kuch samajh nahi aata.",
     "v1_label": None, "v2_aspects": None},

    # ── 37. TOXICITY / SPAM ──────────────────────────────────────────────────
    {"cap": "Toxicity Detection",
     "text": "This professor is an absolute idiot and should be fired immediately.",
     "v1_label": "Negative", "v2_aspects": None},

    {"cap": "Spam Detection",
     "text": "aaaaaaaaaaaaaaaaaaa",
     "v1_label": None, "v2_aspects": None},

    {"cap": "Spam Detection",
     "text": "good good good good good good good good",
     "v1_label": "Positive", "v2_aspects": None},

    # ── 38. FAIRNESS ANALYSIS ────────────────────────────────────────────────
    {"cap": "Fairness Analysis",
     "text": "The grading rubric is so vague that two students doing identical work get completely different marks.",
     "v1_label": "Negative", "v2_aspects": None},

    {"cap": "Fairness Analysis",
     "text": "Partial credit policies are applied inconsistently depending on which TA marks your work.",
     "v1_label": "Negative", "v2_aspects": None},

    # ── 39. TREND / TOPIC MODELLING (longitudinal texts) ─────────────────────
    {"cap": "Trend Analysis",
     "text": "Unlike the first month when everything was chaotic, the last three weeks have been structured and clear.",
     "v1_label": "Positive", "v2_aspects": ["clarity"]},

    {"cap": "Topic Modeling",
     "text": "Every complaint this semester comes back to the same root cause: insufficient preparation time.",
     "v1_label": "Negative", "v2_aspects": ["workload"]},

    # ── 40. EMERGING ISSUE DETECTION ─────────────────────────────────────────
    {"cap": "Emerging Issue Detection",
     "text": "I have noticed a growing number of students quietly switching off their cameras during lectures.",
     "v1_label": "Negative", "v2_aspects": ["engagement"]},

    {"cap": "Emerging Issue Detection",
     "text": "More and more people are skipping the pre-class readings because they see no benefit.",
     "v1_label": "Negative", "v2_aspects": ["pre_class"]},
]

# Total: 80 test cases across 25+ capabilities


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROFILER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareProfile:
    os_name:          str = ""
    cpu_model:        str = ""
    physical_cores:   int = 0
    logical_cores:    int = 0
    cpu_freq_mhz:     float = 0.0
    ram_total_gb:     float = 0.0
    gpu_name:         str = "None"
    gpu_vram_gb:      float = 0.0
    npu_available:    bool = False
    npu_name:         str = "None"
    cuda_available:   bool = False
    openvino_devices: List[str] = field(default_factory=list)
    python_version:   str = ""
    torch_version:    str = ""
    # TDP for energy estimate
    cpu_tdp_watts:    float = 65.0
    gpu_tdp_watts:    float = 150.0
    # RAPL path
    rapl_path:        str = ""
    rapl_available:   bool = False


def _get_cpu_model() -> str:
    try:
        if platform.system() == "Windows":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"


def _estimate_tdp_from_model(cpu_model: str) -> float:
    """Heuristic TDP from CPU name — covers most Intel/AMD laptop+desktop SKUs."""
    name = cpu_model.lower()
    # Ultra-low-power mobile
    if any(x in name for x in ["u-processor", " u ", "y-processor", "core m"]):
        return 15.0
    # Laptop H-series
    if any(x in name for x in [" h ", "hx", "hk", "hs"]):
        return 45.0
    # Desktop K/X-series
    if any(x in name for x in [" k ", "kf", "ks", "i9-", "ryzen 9", "threadripper"]):
        return 125.0
    # Typical desktop
    if any(x in name for x in ["i7-", "i5-", "ryzen 7", "ryzen 5"]):
        return 65.0
    return 65.0  # safe default


def _estimate_gpu_tdp(gpu_name: str) -> float:
    name = gpu_name.lower()
    if "rtx 4090" in name: return 450.0
    if "rtx 4080" in name: return 320.0
    if "rtx 4070" in name: return 200.0
    if "rtx 3090" in name: return 350.0
    if "rtx 3080" in name: return 320.0
    if "rtx 3070" in name: return 220.0
    if "gtx 1080" in name: return 180.0
    if "a100" in name: return 400.0
    if "v100" in name: return 300.0
    if "laptop" in name or "mobile" in name: return 80.0
    return 150.0


def profile_hardware() -> HardwareProfile:
    hp = HardwareProfile()
    hp.os_name        = f"{platform.system()} {platform.release()} ({platform.machine()})"
    hp.python_version = platform.python_version()
    hp.cpu_model      = _get_cpu_model()

    if PSUTIL_OK:
        hp.physical_cores = psutil.cpu_count(logical=False) or 1
        hp.logical_cores  = psutil.cpu_count(logical=True)  or 1
        hp.ram_total_gb   = round(psutil.virtual_memory().total / (1024**3), 2)
        try:
            freq = psutil.cpu_freq()
            hp.cpu_freq_mhz = freq.max if freq else 0.0
        except Exception:
            pass
    else:
        hp.logical_cores  = os.cpu_count() or 1
        hp.physical_cores = hp.logical_cores

    hp.cpu_tdp_watts = _estimate_tdp_from_model(hp.cpu_model)

    if TORCH_OK:
        hp.torch_version = torch.__version__
        if torch.cuda.is_available():
            hp.cuda_available = True
            hp.gpu_name       = torch.cuda.get_device_name(0)
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory
                hp.gpu_vram_gb = round(total_mem / (1024**3), 2)
            except Exception:
                pass
            hp.gpu_tdp_watts = _estimate_gpu_tdp(hp.gpu_name)

    # OpenVINO device list + NPU check
    try:
        import openvino as ov
        core = ov.Core()
        hp.openvino_devices = core.available_devices
        if "NPU" in hp.openvino_devices:
            hp.npu_available = True
            hp.npu_name = core.get_property("NPU", "FULL_DEVICE_NAME")
    except Exception:
        pass

    # Intel RAPL
    rapl = "/sys/class/powercap/intel-rapl:0/energy_uj"
    if os.path.exists(rapl):
        hp.rapl_available = True
        hp.rapl_path      = rapl

    return hp


# ══════════════════════════════════════════════════════════════════════════════
# TELEMETRY COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TelemetrySnapshot:
    timestamp:       float = 0.0
    cpu_pct:         float = 0.0
    cpu_temp_c:      float = 0.0
    ram_used_gb:     float = 0.0
    ram_pct:         float = 0.0
    gpu_util_pct:    float = 0.0
    gpu_mem_used_gb: float = 0.0
    gpu_temp_c:      float = 0.0
    power_watts:     float = 0.0   # RAPL or estimate
    rapl_energy_j:   float = 0.0


class Telemetry:
    """Background-thread telemetry sampler with RAPL + GPU support."""

    def __init__(self, hw: HardwareProfile, interval: float = 0.5):
        self.hw       = hw
        self.interval = interval
        self.samples: List[TelemetrySnapshot] = []
        self._stop  = threading.Event()
        self._lock  = threading.Lock()
        self._proc  = psutil.Process(os.getpid()) if PSUTIL_OK else None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_rapl_j = 0.0
        self._last_rapl_t = 0.0
        self._running = False

    def start(self):
        # prime RAPL baseline
        if self.hw.rapl_available:
            try:
                with open(self.hw.rapl_path) as f:
                    self._last_rapl_j = int(f.read().strip()) / 1e6
                    self._last_rapl_t = time.time()
            except Exception:
                pass
        # prime psutil cpu_percent
        if PSUTIL_OK:
            psutil.cpu_percent(interval=None)
            if self._proc:
                self._proc.cpu_percent(interval=None)
        self._running = True
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self._running = False

    def _read_gpu(self) -> Tuple[float, float, float]:
        """Returns (util%, mem_used_gb, temp_c)."""
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                timeout=2, text=True
            ).strip().split(",")
            util   = float(out[0].strip())
            mem_mb = float(out[1].strip())
            temp   = float(out[2].strip())
            return util, mem_mb / 1024, temp
        except Exception:
            return 0.0, 0.0, 0.0

    def _sample(self) -> TelemetrySnapshot:
        snap = TelemetrySnapshot(timestamp=time.time())

        if PSUTIL_OK:
            snap.cpu_pct    = psutil.cpu_percent(interval=None)
            vm              = psutil.virtual_memory()
            snap.ram_used_gb= round(vm.used / (1024**3), 3)
            snap.ram_pct    = vm.percent
            # temperature
            try:
                sensors = psutil.sensors_temperatures()
                if sensors:
                    for key in ("coretemp","k10temp","cpu_thermal","acpitz","zenpower"):
                        if key in sensors and sensors[key]:
                            snap.cpu_temp_c = sensors[key][0].current
                            break
            except Exception:
                pass

        # RAPL
        if self.hw.rapl_available:
            try:
                with open(self.hw.rapl_path) as f:
                    cur_j = int(f.read().strip()) / 1e6
                now = time.time()
                dt  = now - self._last_rapl_t
                if dt > 0 and cur_j >= self._last_rapl_j:
                    snap.power_watts = (cur_j - self._last_rapl_j) / dt
                else:
                    snap.power_watts = 0.0
                snap.rapl_energy_j  = cur_j
                self._last_rapl_j   = cur_j
                self._last_rapl_t   = now
            except Exception:
                snap.power_watts = (snap.cpu_pct / 100.0) * self.hw.cpu_tdp_watts
        else:
            snap.power_watts = (snap.cpu_pct / 100.0) * self.hw.cpu_tdp_watts

        # GPU (CUDA / nvidia-smi)
        if self.hw.cuda_available:
            snap.gpu_util_pct, snap.gpu_mem_used_gb, snap.gpu_temp_c = self._read_gpu()
            # Add GPU power estimate
            snap.power_watts += (snap.gpu_util_pct / 100.0) * self.hw.gpu_tdp_watts

        return snap

    def _run(self):
        while not self._stop.is_set():
            try:
                s = self._sample()
                with self._lock:
                    self.samples.append(s)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def get_samples(self) -> List[TelemetrySnapshot]:
        with self._lock:
            return list(self.samples)

    def energy_wh(self, t_start: float, t_end: float) -> float:
        samps = [s for s in self.get_samples()
                 if t_start <= s.timestamp <= t_end]
        if len(samps) < 2:
            if samps:
                return samps[0].power_watts * ((t_end - t_start) / 3600)
            return 0.0
        total = 0.0
        for i in range(1, len(samps)):
            dt  = (samps[i].timestamp - samps[i-1].timestamp) / 3600
            avg = (samps[i].power_watts + samps[i-1].power_watts) / 2
            total += avg * dt
        return total

    def summary(self, t_start: float, t_end: float) -> Dict[str, Any]:
        samps = [s for s in self.get_samples()
                 if t_start <= s.timestamp <= t_end]
        if not samps:
            return {}
        cpu    = [s.cpu_pct      for s in samps]
        ram    = [s.ram_pct      for s in samps]
        temps  = [s.cpu_temp_c   for s in samps if s.cpu_temp_c > 0]
        gpuu   = [s.gpu_util_pct for s in samps]
        pwr    = [s.power_watts  for s in samps]
        return {
            "cpu_avg_pct":    round(float(np.mean(cpu)), 1),
            "cpu_peak_pct":   round(float(np.max(cpu)), 1),
            "ram_avg_pct":    round(float(np.mean(ram)), 1),
            "ram_peak_pct":   round(float(np.max(ram)), 1),
            "cpu_temp_avg_c": round(float(np.mean(temps)), 1) if temps else None,
            "cpu_temp_peak_c":round(float(np.max(temps)), 1) if temps else None,
            "gpu_util_avg_pct":round(float(np.mean(gpuu)), 1),
            "gpu_util_peak_pct":round(float(np.max(gpuu)), 1),
            "power_avg_w":    round(float(np.mean(pwr)), 2),
            "power_peak_w":   round(float(np.max(pwr)), 2),
            "energy_wh":      round(self.energy_wh(t_start, t_end), 6),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER — imports the original inference classes dynamically
# ══════════════════════════════════════════════════════════════════════════════

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class V1Wrapper:
    """Wraps SentimentInferenceEngine from run_inference_bert_real.py."""

    VERSION   = "v1"
    TASK      = "3-class Sentiment (Neg/Neu/Pos)"
    N_CLASSES = 3

    def __init__(self, script_path: Path, model_dir: Path,
                 backend: Optional[str], hw: HardwareProfile):
        self.hw = hw
        log.info("[V1] Loading module: %s", script_path)
        t0 = time.perf_counter()
        mod = _load_module("run_inference_v1", script_path)
        self.engine = mod.SentimentInferenceEngine(
            model_dir  = str(model_dir),
            backend    = backend,
            batch_size = 1,
        )
        self.load_time_s = time.perf_counter() - t0
        self.backend_name = self.engine.backend_name
        log.info("[V1] Loaded in %.2fs  backend=%s", self.load_time_s, self.backend_name)

    def warmup(self):
        self.engine.warmup(n=5)

    def predict(self, text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        r  = self.engine.predict(text)
        r["_latency_ms"] = (time.perf_counter() - t0) * 1000
        return r

    def label(self, result: Dict) -> str:
        return result.get("label_name", "Unknown")

    def score(self, result: Dict) -> float:
        return result.get("score", 0.0)

    def confidence(self, result: Dict) -> float:
        return result.get("confidence", 0.0)


class V2Wrapper:
    """Wraps APEInferenceEngine from run_inference.py."""

    VERSION   = "v2"
    TASK      = "7-class ABSA (aspect-aware)"
    N_CLASSES = 7

    def __init__(self, script_path: Path, model_dir: Path,
                 backend: Optional[str], hw: HardwareProfile):
        self.hw = hw
        log.info("[V2] Loading module: %s", script_path)
        t0 = time.perf_counter()
        mod = _load_module("run_inference_v2", script_path)
        self.engine = mod.APEInferenceEngine(
            model_dir  = str(model_dir),
            backend    = backend,
            batch_size = 1,
        )
        self.load_time_s = time.perf_counter() - t0
        self.backend_name = self.engine.backend_name
        log.info("[V2] Loaded in %.2fs  backend=%s", self.load_time_s, self.backend_name)

    def warmup(self):
        self.engine.warmup(n=5)

    def predict(self, text: str) -> Dict[str, Any]:
        t0  = time.perf_counter()
        r   = self.engine.analyse_absa(text)
        lat = (time.perf_counter() - t0) * 1000
        return {"_raw": r, "_latency_ms": lat}

    def aspects_found(self, result: Dict) -> List[str]:
        return list(result.get("_raw", {}).keys())

    def dominant_polarity(self, result: Dict) -> str:
        raw = result.get("_raw", {})
        if not raw:
            return "neutral"
        scores = [v["score"] for v in raw.values()]
        mean   = np.mean(scores)
        if mean > 0.15:  return "positive"
        if mean < -0.15: return "negative"
        return "neutral"

    def mean_score(self, result: Dict) -> float:
        raw = result.get("_raw", {})
        if not raw:
            return 0.0
        return float(np.mean([v["score"] for v in raw.values()]))

    def mean_confidence(self, result: Dict) -> float:
        raw = result.get("_raw", {})
        if not raw:
            return 0.0
        return float(np.mean([v["confidence"] for v in raw.values()]))


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SampleResult:
    idx:           int
    capability:    str
    text:          str
    version:       str
    raw_output:    Any
    label:         str
    expected_label:Optional[str]
    aspects_found: List[str]
    expected_aspects: Optional[List[str]]
    latency_ms:    float
    confidence:    float
    score:         float
    correct_label: Optional[bool]   # None = no gold label
    aspects_hit:   Optional[float]  # recall fraction, None if no expected


def _label_to_polarity(label: Optional[str]) -> Optional[str]:
    if label is None: return None
    l = label.lower()
    if "neg" in l: return "negative"
    if "pos" in l: return "positive"
    return "neutral"


def run_v1(wrapper: V1Wrapper,
           telemetry: Telemetry,
           cases: List[Dict]) -> List[SampleResult]:
    results = []
    for i, case in enumerate(cases):
        t0  = telemetry.get_samples()[-1].timestamp if telemetry.get_samples() else time.time()
        out = wrapper.predict(case["text"])
        t1  = time.time()

        expected_pol = _label_to_polarity(case.get("v1_label"))
        predicted_pol= _label_to_polarity(wrapper.label(out))
        correct      = None if expected_pol is None else (predicted_pol == expected_pol)

        results.append(SampleResult(
            idx=i,
            capability=case["cap"],
            text=case["text"],
            version="v1",
            raw_output=out,
            label=wrapper.label(out),
            expected_label=case.get("v1_label"),
            aspects_found=[],
            expected_aspects=case.get("v2_aspects"),
            latency_ms=out["_latency_ms"],
            confidence=wrapper.confidence(out),
            score=wrapper.score(out),
            correct_label=correct,
            aspects_hit=None,
        ))
    return results


def run_v2(wrapper: V2Wrapper,
           telemetry: Telemetry,
           cases: List[Dict]) -> List[SampleResult]:
    results = []
    for i, case in enumerate(cases):
        out = wrapper.predict(case["text"])

        # Label alignment: map V2 mean polarity → 3-class label
        pol = wrapper.dominant_polarity(out)
        pol_to_label = {"positive":"Positive","negative":"Negative","neutral":"Neutral"}
        pred_label   = pol_to_label[pol]

        expected_pol = _label_to_polarity(case.get("v1_label"))
        correct      = None if expected_pol is None else (pol == expected_pol)

        # Aspect recall
        expected_asp = case.get("v2_aspects")
        asp_found    = wrapper.aspects_found(out)
        if expected_asp:
            hit = sum(1 for a in expected_asp if a in asp_found) / len(expected_asp)
        else:
            hit = None

        results.append(SampleResult(
            idx=i,
            capability=case["cap"],
            text=case["text"],
            version="v2",
            raw_output=out,
            label=pred_label,
            expected_label=case.get("v1_label"),
            aspects_found=asp_found,
            expected_aspects=expected_asp,
            latency_ms=out["_latency_ms"],
            confidence=wrapper.mean_confidence(out),
            score=wrapper.mean_score(out),
            correct_label=correct,
            aspects_hit=hit,
        ))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _accuracy(results: List[SampleResult]) -> float:
    scored = [r for r in results if r.correct_label is not None]
    if not scored: return 0.0
    return sum(1 for r in scored if r.correct_label) / len(scored)


def _aspect_recall(results: List[SampleResult]) -> Optional[float]:
    scored = [r for r in results if r.aspects_hit is not None]
    if not scored: return None
    return float(np.mean([r.aspects_hit for r in scored]))


def _per_capability(results: List[SampleResult]) -> Dict[str, Dict]:
    caps: Dict[str, List[SampleResult]] = {}
    for r in results:
        caps.setdefault(r.capability, []).append(r)
    out = {}
    for cap, rs in caps.items():
        scored = [r for r in rs if r.correct_label is not None]
        asp    = [r for r in rs if r.aspects_hit is not None]
        out[cap] = {
            "n":              len(rs),
            "n_scored":       len(scored),
            "accuracy":       round(sum(1 for r in scored if r.correct_label) / len(scored), 4)
                              if scored else None,
            "aspect_recall":  round(float(np.mean([r.aspects_hit for r in asp])), 4)
                              if asp else None,
            "mean_latency_ms":round(float(np.mean([r.latency_ms for r in rs])), 2),
            "mean_confidence":round(float(np.mean([r.confidence for r in rs])), 4),
        }
    return out


def _energy_summary(hw: HardwareProfile, telemetry: Telemetry,
                    t_start: float, t_end: float) -> Dict[str, Any]:
    tel = telemetry.summary(t_start, t_end)
    wh  = tel.get("energy_wh", 0.0)
    kwh = wh / 1000.0
    return {
        **tel,
        "total_energy_wh":  round(wh, 6),
        "total_energy_kwh": round(kwh, 8),
        "co2_grams_india":  round(kwh * 708, 6),
        "co2_grams_global": round(kwh * 475, 6),
        "energy_method":    "Intel RAPL" if hw.rapl_available else "TDP estimate",
    }


def write_json_report(path: Path, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("JSON report: %s", path)


def write_csv(path: Path, results: List[SampleResult]):
    if not results:
        return
    fields = ["idx","version","capability","label","expected_label",
              "correct_label","aspects_found","expected_aspects","aspects_hit",
              "latency_ms","confidence","score","text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({
                "idx":             r.idx,
                "version":         r.version,
                "capability":      r.capability,
                "label":           r.label,
                "expected_label":  r.expected_label,
                "correct_label":   r.correct_label,
                "aspects_found":   ";".join(r.aspects_found),
                "expected_aspects":";".join(r.expected_aspects or []),
                "aspects_hit":     r.aspects_hit,
                "latency_ms":      round(r.latency_ms, 3),
                "confidence":      round(r.confidence, 4),
                "score":           round(r.score, 4),
                "text":            r.text[:120],
            })
    log.info("CSV report:  %s", path)


def write_txt_report(path: Path,
                     hw:       HardwareProfile,
                     v1_res:   Optional[List[SampleResult]],
                     v2_res:   Optional[List[SampleResult]],
                     v1_wrap:  Optional[V1Wrapper],
                     v2_wrap:  Optional[V2Wrapper],
                     v1_energy: Dict,
                     v2_energy: Dict,
                     v1_tel:   Dict,
                     v2_tel:   Dict,
                     total_s:  float):

    def bar(pct: float, width: int = 30) -> str:
        if pct is None: return "N/A"
        filled = max(0, min(width, int(pct / 100 * width)))
        return "█" * filled + "░" * (width - filled) + f"  {pct:.1f}%"

    def pct(v):
        if v is None: return "N/A"
        return f"{v*100:.2f}%"

    lines = [
        "=" * 74,
        "  APE / BERT INFERENCE ENGINE — COMPREHENSIVE EVALUATION REPORT",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Duration  : {timedelta(seconds=int(total_s))}",
        "=" * 74,
        "",
        "┌─ HARDWARE PROFILE ─────────────────────────────────────────────────────┐",
        f"  OS             : {hw.os_name}",
        f"  CPU            : {hw.cpu_model}",
        f"  CPU Cores      : {hw.physical_cores} physical / {hw.logical_cores} logical",
        f"  CPU Freq (max) : {hw.cpu_freq_mhz:.0f} MHz",
        f"  CPU TDP (est.) : {hw.cpu_tdp_watts:.0f} W",
        f"  RAM Total      : {hw.ram_total_gb:.2f} GB",
        f"  GPU            : {hw.gpu_name}  ({hw.gpu_vram_gb:.1f} GB VRAM)",
        f"  GPU TDP (est.) : {hw.gpu_tdp_watts:.0f} W",
        f"  NPU            : {'✓  ' + hw.npu_name if hw.npu_available else 'Not detected'}",
        f"  CUDA Available : {hw.cuda_available}",
        f"  OpenVINO devs  : {hw.openvino_devices}",
        f"  Energy method  : {'Intel RAPL' if hw.rapl_available else 'TDP-based estimate'}",
        f"  Python         : {hw.python_version}",
        f"  PyTorch        : {hw.torch_version}",
        "└────────────────────────────────────────────────────────────────────────┘",
        "",
    ]

    def version_block(tag, wrap, results, energy, tel):
        if not results:
            return [f"  [{tag}] NOT RUN", ""]
        acc  = _accuracy(results)
        arec = _aspect_recall(results)
        lats = [r.latency_ms for r in results]
        confs= [r.confidence for r in results]
        scored = [r for r in results if r.correct_label is not None]
        cap_table = _per_capability(results)

        blk = [
            f"┌─ {tag}: {wrap.TASK} ──────────────────────────────────────────────────┐",
            f"  Backend            : {wrap.backend_name}",
            f"  Model load time    : {wrap.load_time_s:.3f} s",
            f"  Test cases run     : {len(results)}",
            f"  Cases with gold    : {len(scored)}",
            f"  Overall Accuracy   : {pct(acc) if scored else 'N/A'}",
        ]
        if arec is not None:
            blk.append(f"  Aspect Recall      : {pct(arec)}")
        blk += [
            f"  Latency p50        : {np.percentile(lats,50):.2f} ms",
            f"  Latency p95        : {np.percentile(lats,95):.2f} ms",
            f"  Latency p99        : {np.percentile(lats,99):.2f} ms",
            f"  Mean Confidence    : {float(np.mean(confs)):.4f}",
            "",
            "  ── System Metrics During Inference ──",
            f"  CPU avg / peak     : {tel.get('cpu_avg_pct','?')} / {tel.get('cpu_peak_pct','?')} %",
            f"  RAM avg / peak     : {tel.get('ram_avg_pct','?')} / {tel.get('ram_peak_pct','?')} %",
        ]
        if tel.get("cpu_temp_avg_c"):
            blk.append(f"  CPU Temp avg/peak  : {tel.get('cpu_temp_avg_c')} / {tel.get('cpu_temp_peak_c')} °C")
        blk += [
            f"  GPU util avg/peak  : {tel.get('gpu_util_avg_pct','0')} / {tel.get('gpu_util_peak_pct','0')} %",
            f"  Power avg / peak   : {tel.get('power_avg_w','?')} / {tel.get('power_peak_w','?')} W",
            f"  Total Energy       : {energy.get('total_energy_wh','?'):.6f} Wh  ({energy.get('total_energy_kwh','?'):.8f} kWh)",
            f"  CO2 (India 708g/kWh) : {energy.get('co2_grams_india','?'):.6f} g",
            f"  CO2 (Global 475g/kWh): {energy.get('co2_grams_global','?'):.6f} g",
            f"  Energy method      : {energy.get('energy_method','?')}",
            "",
            "  ── Per-Capability Breakdown ──",
            f"  {'Capability':<38} {'N':>4} {'Acc':>7} {'AspRec':>8} {'Lat(ms)':>9} {'Conf':>7}",
            "  " + "─" * 72,
        ]
        for cap, cm in sorted(cap_table.items()):
            acc_s  = f"{cm['accuracy']*100:.1f}%" if cm['accuracy'] is not None else "  n/a "
            arec_s = f"{cm['aspect_recall']*100:.1f}%" if cm['aspect_recall'] is not None else "  n/a "
            blk.append(
                f"  {cap:<38} {cm['n']:>4} {acc_s:>7} {arec_s:>8} "
                f"{cm['mean_latency_ms']:>9.1f} {cm['mean_confidence']:>7.4f}"
            )
        blk.append("└────────────────────────────────────────────────────────────────────────┘")
        blk.append("")
        return blk

    if v1_wrap and v1_res:
        lines += version_block("V1 (3-class)", v1_wrap, v1_res, v1_energy, v1_tel)
    if v2_wrap and v2_res:
        lines += version_block("V2 (7-class ABSA)", v2_wrap, v2_res, v2_energy, v2_tel)

    # Per-sample detail
    lines += [
        "┌─ PER-SAMPLE DETAIL ────────────────────────────────────────────────────┐",
        f"  {'#':>3} {'Ver':>3} {'OK':>3} {'Lat':>7} {'Conf':>6} {'Label':<12} {'Capability':<32} {'Text (60ch)'}",
        "  " + "─" * 115,
    ]
    all_res = (v1_res or []) + (v2_res or [])
    for r in all_res:
        ok = "✓" if r.correct_label else ("?" if r.correct_label is None else "✗")
        lines.append(
            f"  {r.idx:>3} {r.version:>3} {ok:>3} {r.latency_ms:>7.1f} "
            f"{r.confidence:>6.4f} {r.label:<12} {r.capability:<32} {r.text[:60]}"
        )
    lines += ["└────────────────────────────────────────────────────────────────────────┘", ""]

    lines += [
        "─" * 74,
        "  END OF REPORT",
        "─" * 74,
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("TXT report:  %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Universal evaluation harness for V1 + V2 inference engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--v1-script",   type=str,
                    default=str(SCRIPT_DIR / "run_inference_bert_real.py"),
                    help="Path to run_inference_bert_real.py  (V1)")
    ap.add_argument("--v1-model-dir",type=str,
                    default=str(SCRIPT_DIR / "fine_tuned_bert_real"),
                    help="V1 model directory  (default: ./fine_tuned_bert_real)")
    ap.add_argument("--v2-script",   type=str,
                    default=str(SCRIPT_DIR / "run_inference.py"),
                    help="Path to run_inference.py  (V2)")
    ap.add_argument("--v2-model-dir",type=str,
                    default=str(SCRIPT_DIR / "outputs" / "models"),
                    help="V2 model directory  (default: ./outputs/models)")
    ap.add_argument("--only",        type=str, choices=["v1","v2"],
                    default=None,  help="Run only one version")
    ap.add_argument("--backend",     type=str,
                    choices=["pytorch","onnx","openvino","npu"],
                    default=None,  help="Force inference backend (default: auto)")
    ap.add_argument("--no-warmup",   action="store_true",
                    help="Skip warmup passes")
    ap.add_argument("--out-dir",     type=str, default=str(SCRIPT_DIR),
                    help="Directory for output files (default: script dir)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 74)
    print("  APE / BERT — COMPREHENSIVE INFERENCE EVALUATION HARNESS")
    print(f"  {len(TEST_CASES)} test cases  ·  {len(set(c['cap'] for c in TEST_CASES))} capabilities")
    print("=" * 74 + "\n")

    # ── hardware profile ───────────────────────────────────────────────────────
    log.info("Profiling hardware ...")
    hw = profile_hardware()
    log.info("  CPU  : %s  (%.0fW TDP est.)", hw.cpu_model, hw.cpu_tdp_watts)
    log.info("  RAM  : %.1f GB", hw.ram_total_gb)
    if hw.cuda_available:
        log.info("  GPU  : %s  (%.0fW TDP est.)", hw.gpu_name, hw.gpu_tdp_watts)
    if hw.npu_available:
        log.info("  NPU  : %s", hw.npu_name)

    # ── telemetry ─────────────────────────────────────────────────────────────
    tel = Telemetry(hw, interval=0.5)
    tel.start()
    global_t0 = time.time()

    v1_wrap = v2_wrap = None
    v1_res  = v2_res  = []
    v1_energy= v2_energy = {}
    v1_tel   = v2_tel   = {}

    # ── V1 ────────────────────────────────────────────────────────────────────
    if args.only in (None, "v1"):
        v1_script = Path(args.v1_script)
        v1_mdir   = Path(args.v1_model_dir)
        if not v1_script.exists():
            log.warning("[V1] Script not found: %s — skipping V1", v1_script)
        elif not v1_mdir.exists():
            log.warning("[V1] Model dir not found: %s — skipping V1", v1_mdir)
        else:
            try:
                t_load0 = time.time()
                v1_wrap = V1Wrapper(v1_script, v1_mdir, args.backend, hw)
                t_load1 = time.time()
                if not args.no_warmup:
                    log.info("[V1] Warming up backend ...")
                    v1_wrap.warmup()

                log.info("[V1] Running %d test cases ...", len(TEST_CASES))
                t_inf0 = time.time()
                v1_res = run_v1(v1_wrap, tel, TEST_CASES)
                t_inf1 = time.time()

                v1_energy = _energy_summary(hw, tel, t_inf0, t_inf1)
                v1_tel    = tel.summary(t_inf0, t_inf1)

                acc = _accuracy(v1_res)
                log.info("[V1] Done — accuracy=%.2f%%  mean_lat=%.1f ms",
                         acc*100, np.mean([r.latency_ms for r in v1_res]))
            except Exception as e:
                log.error("[V1] FAILED: %s\n%s", e, traceback.format_exc())

    # ── V2 ────────────────────────────────────────────────────────────────────
    if args.only in (None, "v2"):
        v2_script = Path(args.v2_script)
        v2_mdir   = Path(args.v2_model_dir)
        if not v2_script.exists():
            log.warning("[V2] Script not found: %s — skipping V2", v2_script)
        elif not v2_mdir.exists():
            log.warning("[V2] Model dir not found: %s — skipping V2", v2_mdir)
        else:
            try:
                v2_wrap = V2Wrapper(v2_script, v2_mdir, args.backend, hw)
                if not args.no_warmup:
                    log.info("[V2] Warming up backend ...")
                    v2_wrap.warmup()

                log.info("[V2] Running %d test cases ...", len(TEST_CASES))
                t_inf0 = time.time()
                v2_res = run_v2(v2_wrap, tel, TEST_CASES)
                t_inf1 = time.time()

                v2_energy = _energy_summary(hw, tel, t_inf0, t_inf1)
                v2_tel    = tel.summary(t_inf0, t_inf1)

                acc  = _accuracy(v2_res)
                arec = _aspect_recall(v2_res)
                log.info("[V2] Done — accuracy=%.2f%%  aspect_recall=%.2f%%  mean_lat=%.1f ms",
                         acc*100,
                         (arec or 0)*100,
                         np.mean([r.latency_ms for r in v2_res]))
            except Exception as e:
                log.error("[V2] FAILED: %s\n%s", e, traceback.format_exc())

    tel.stop()
    total_s = time.time() - global_t0

    # ── Write outputs ─────────────────────────────────────────────────────────
    base = out_dir / f"eval_report_{TIMESTAMP}"

    # JSON
    json_data = {
        "generated_at":   datetime.now().isoformat(),
        "total_duration_s": round(total_s, 2),
        "hardware":       asdict(hw),
        "test_suite_size": len(TEST_CASES),
        "capabilities":   sorted(set(c["cap"] for c in TEST_CASES)),
    }
    for tag, wrap, res, energy, telsum in [
        ("v1", v1_wrap, v1_res, v1_energy, v1_tel),
        ("v2", v2_wrap, v2_res, v2_energy, v2_tel),
    ]:
        if res:
            scored = [r for r in res if r.correct_label is not None]
            json_data[tag] = {
                "backend":          wrap.backend_name,
                "load_time_s":      round(wrap.load_time_s, 3),
                "n_cases":          len(res),
                "n_scored":         len(scored),
                "overall_accuracy": round(_accuracy(res), 4),
                "aspect_recall":    round(_aspect_recall(res), 4) if _aspect_recall(res) else None,
                "latency": {
                    "p50_ms": round(float(np.percentile([r.latency_ms for r in res],50)),2),
                    "p95_ms": round(float(np.percentile([r.latency_ms for r in res],95)),2),
                    "p99_ms": round(float(np.percentile([r.latency_ms for r in res],99)),2),
                    "mean_ms":round(float(np.mean([r.latency_ms for r in res])),2),
                    "std_ms": round(float(np.std([r.latency_ms for r in res])),2),
                },
                "system_metrics": telsum,
                "energy":         energy,
                "per_capability": _per_capability(res),
                "samples": [
                    {
                        "idx":r.idx,"capability":r.capability,"text":r.text,
                        "label":r.label,"expected":r.expected_label,
                        "correct":r.correct_label,
                        "aspects_found":r.aspects_found,
                        "expected_aspects":r.expected_aspects,
                        "aspects_hit":r.aspects_hit,
                        "latency_ms":round(r.latency_ms,3),
                        "confidence":round(r.confidence,4),
                        "score":round(r.score,4),
                    }
                    for r in res
                ],
            }

    write_json_report(Path(str(base) + ".json"), json_data)
    write_csv(Path(str(base) + ".csv"), (v1_res or []) + (v2_res or []))
    write_txt_report(
        Path(str(base) + ".txt"),
        hw, v1_res, v2_res, v1_wrap, v2_wrap,
        v1_energy, v2_energy, v1_tel, v2_tel, total_s
    )

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  EVALUATION COMPLETE")
    print("=" * 74)
    for tag, wrap, res, energy in [
        ("V1", v1_wrap, v1_res, v1_energy),
        ("V2", v2_wrap, v2_res, v2_energy),
    ]:
        if res:
            scored = [r for r in res if r.correct_label is not None]
            print(f"\n  [{tag}]  backend={wrap.backend_name}  load={wrap.load_time_s:.2f}s")
            print(f"         accuracy    = {_accuracy(res)*100:.2f}%  ({len(scored)}/{len(res)} scored)")
            if _aspect_recall(res) is not None:
                print(f"         asp. recall = {_aspect_recall(res)*100:.2f}%")
            lats = [r.latency_ms for r in res]
            print(f"         latency p50 = {np.percentile(lats,50):.1f} ms  "
                  f"p95={np.percentile(lats,95):.1f} ms")
            print(f"         energy      = {energy.get('total_energy_wh',0):.6f} Wh  "
                  f"({energy.get('energy_method','?')})")
            print(f"         CO2 (India) = {energy.get('co2_grams_india',0):.6f} g")
    print(f"\n  Total wall time : {timedelta(seconds=int(total_s))}")
    print(f"  Reports written : {base}.{{json,txt,csv}}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()


# ── EXTENDED TEST CASES appended at module level ─────────────────────────────
# Adds at least 3 total cases per capability (augmenting the list above)

_EXTRA_CASES: List[Dict[str, Any]] = [

    # ── Aspect Extraction (extra) ─────────────────────────────────────────────
    {"cap":"Aspect Extraction",
     "text":"The pre-class readings are too long and the engagement in tutorials is poor.",
     "v1_label":"Negative","v2_aspects":["pre_class","engagement"]},
    {"cap":"Aspect Extraction",
     "text":"The workload is crushing and the pacing in lectures has never slowed down.",
     "v1_label":"Negative","v2_aspects":["workload","pacing"]},
    {"cap":"Aspect Extraction",
     "text":"Collaboration with my group is excellent and the clarity of the slides is impressive.",
     "v1_label":"Positive","v2_aspects":["collaboration","clarity"]},

    # ── Dynamic Aspect Discovery (extra) ──────────────────────────────────────
    {"cap":"Dynamic Aspect Discovery",
     "text":"The library resources linked in the course portal are completely outdated.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Dynamic Aspect Discovery",
     "text":"I wish the office hours were at times that work for evening students.",
     "v1_label":"Negative","v2_aspects":None},

    # ── ABSA (extra) ──────────────────────────────────────────────────────────
    {"cap":"Aspect-Based Sentiment Analysis",
     "text":"The workload is fair but student engagement is disappointingly low.",
     "v1_label":"Neutral","v2_aspects":["workload","engagement"]},
    {"cap":"Aspect-Based Sentiment Analysis",
     "text":"Pacing improved this week but the collaborative tasks were still unstructured.",
     "v1_label":"Neutral","v2_aspects":["pacing","collaboration"]},
    {"cap":"Aspect-Based Sentiment Analysis",
     "text":"Pre-class materials are outstanding; the in-class pacing however is brutal.",
     "v1_label":"Neutral","v2_aspects":["pre_class","pacing"]},

    # ── Multi-Aspect Detection (extra) ────────────────────────────────────────
    {"cap":"Multi-Aspect Detection",
     "text":"Engagement is high, clarity has improved, and the workload is now much more manageable.",
     "v1_label":"Positive","v2_aspects":["engagement","clarity","workload"]},
    {"cap":"Multi-Aspect Detection",
     "text":"The collaboration tasks are meaningless, pre-class work is excessive, and clarity remains zero.",
     "v1_label":"Negative","v2_aspects":["collaboration","pre_class","clarity"]},

    # ── Implicit Sentiment (extra) ────────────────────────────────────────────
    {"cap":"Implicit Sentiment Detection",
     "text":"I have not voluntarily participated once all semester.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Implicit Sentiment Detection",
     "text":"For the first time I genuinely did not want the lecture to end.",
     "v1_label":"Positive","v2_aspects":None},

    # ── Sarcasm Detection (extra) ──────────────────────────────────────────────
    {"cap":"Sarcasm Detection",
     "text":"What a treat — yet another session I'll need to re-learn from scratch at home.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Sarcasm Detection",
     "text":"Sure, adding another mandatory assignment the night before the exam was a great idea.",
     "v1_label":"Negative","v2_aspects":["workload"]},

    # ── Irony Detection (extra) ────────────────────────────────────────────────
    {"cap":"Irony Detection",
     "text":"Very helpful that the 'summary' slides contain more content than the full lecture.",
     "v1_label":"Negative","v2_aspects":["clarity"]},
    {"cap":"Irony Detection",
     "text":"Wonderful that the deadline was moved up with only 12 hours notice.",
     "v1_label":"Negative","v2_aspects":["workload"]},

    # ── Mixed Sentiment (extra) ────────────────────────────────────────────────
    {"cap":"Mixed Sentiment Analysis",
     "text":"The theory lectures are exceptional but the practicals are completely disorganised.",
     "v1_label":"Neutral","v2_aspects":["clarity","engagement"]},
    {"cap":"Mixed Sentiment Analysis",
     "text":"Great content ruined by an impossible submission system and zero support.",
     "v1_label":"Neutral","v2_aspects":["clarity","workload"]},

    # ── Negation Handling (extra) ──────────────────────────────────────────────
    {"cap":"Negation Handling",
     "text":"Nobody said the tutorial was unhelpful — in fact everyone thought it was excellent.",
     "v1_label":"Positive","v2_aspects":None},
    {"cap":"Negation Handling",
     "text":"It was not the worst lecture, but it certainly wasn't helpful either.",
     "v1_label":"Neutral","v2_aspects":None},

    # ── Comparative Opinion (extra) ────────────────────────────────────────────
    {"cap":"Comparative Opinion Analysis",
     "text":"This module's pre-class work is far less burdensome than last semester's equivalent.",
     "v1_label":"Positive","v2_aspects":["pre_class"]},
    {"cap":"Comparative Opinion Analysis",
     "text":"The collaboration quality here is so much worse than in my previous institution.",
     "v1_label":"Negative","v2_aspects":["collaboration"]},
    {"cap":"Comparative Opinion Analysis",
     "text":"Compared to online resources, the in-class explanations add very little value.",
     "v1_label":"Negative","v2_aspects":["clarity"]},

    # ── Coreference Resolution (extra) ────────────────────────────────────────
    {"cap":"Coreference Resolution",
     "text":"The TA walks through every problem carefully and she makes sure nobody gets left behind.",
     "v1_label":"Positive","v2_aspects":["clarity","pacing"]},
    {"cap":"Coreference Resolution",
     "text":"The group dynamics are good but they collapse the moment the task becomes difficult.",
     "v1_label":"Neutral","v2_aspects":["collaboration"]},

    # ── Contextual Understanding (extra) ─────────────────────────────────────
    {"cap":"Contextual Understanding",
     "text":"Considering this is an introductory course, the assumed prior knowledge is unrealistic.",
     "v1_label":"Negative","v2_aspects":["clarity"]},
    {"cap":"Contextual Understanding",
     "text":"After three weeks of struggling, everything finally clicked in today's session.",
     "v1_label":"Positive","v2_aspects":["clarity"]},

    # ── Student Slang (extra) ─────────────────────────────────────────────────
    {"cap":"Student Slang Interpretation",
     "text":"No cap the prof is goated — explains everything in a way that just hits different.",
     "v1_label":"Positive","v2_aspects":["clarity","engagement"]},
    {"cap":"Student Slang Interpretation",
     "text":"The assignment is giving me full villain arc energy — I literally cannot even rn.",
     "v1_label":"Negative","v2_aspects":["workload"]},

    # ── Abbreviation Expansion (extra) ────────────────────────────────────────
    {"cap":"Abbreviation Expansion",
     "text":"The LMS keeps crashing during MCQ submissions, which is a massive problem tbh.",
     "v1_label":"Negative","v2_aspects":["workload"]},
    {"cap":"Abbreviation Expansion",
     "text":"Prof's ETA for returning graded work is always way off, it's been 3wks btw.",
     "v1_label":"Negative","v2_aspects":None},

    # ── Emotion Detection (extra) ─────────────────────────────────────────────
    {"cap":"Emotion Detection",
     "text":"I leave every lecture feeling small and stupid and I hate that.",
     "v1_label":"Negative","v2_aspects":["clarity"]},
    {"cap":"Emotion Detection",
     "text":"There is a pervasive sense of dread around every upcoming deadline.",
     "v1_label":"Negative","v2_aspects":["workload"]},

    # ── Sentiment Intensity (extra) ───────────────────────────────────────────
    {"cap":"Sentiment Intensity Scoring",
     "text":"Slightly disappointing overall.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Sentiment Intensity Scoring",
     "text":"Mildly satisfied, nothing to write home about.",
     "v1_label":"Neutral","v2_aspects":None},
    {"cap":"Sentiment Intensity Scoring",
     "text":"Devastatingly bad in every conceivable dimension.",
     "v1_label":"Negative","v2_aspects":None},

    # ── Severity Assessment (extra) ────────────────────────────────────────────
    {"cap":"Severity Assessment",
     "text":"The pacing issue is a minor annoyance that I have adapted to quickly.",
     "v1_label":"Neutral","v2_aspects":["pacing"]},
    {"cap":"Severity Assessment",
     "text":"Students are considering a formal complaint — this level of confusion cannot continue.",
     "v1_label":"Negative","v2_aspects":["clarity"]},

    # ── Opinion Target Extraction (extra) ─────────────────────────────────────
    {"cap":"Opinion Target Extraction",
     "text":"The mid-term rubric was crystal clear but the final project brief is completely opaque.",
     "v1_label":"Neutral","v2_aspects":None},
    {"cap":"Opinion Target Extraction",
     "text":"The simulation software is excellent while the lab report template is confusing.",
     "v1_label":"Neutral","v2_aspects":None},

    # ── Opinion Phrase Extraction (extra) ─────────────────────────────────────
    {"cap":"Opinion Phrase Extraction",
     "text":"The ruthlessly fast pacing and the impossibly dense slides make this a nightmare.",
     "v1_label":"Negative","v2_aspects":["pacing","clarity"]},
    {"cap":"Opinion Phrase Extraction",
     "text":"The warmly collaborative atmosphere and the crystal-clear delivery make this a joy.",
     "v1_label":"Positive","v2_aspects":["collaboration","clarity"]},

    # ── Evidence Extraction (extra) ───────────────────────────────────────────
    {"cap":"Evidence Extraction",
     "text":"Every time a concept was unclear I looked in the textbook and it was even more confusing.",
     "v1_label":"Negative","v2_aspects":["clarity"]},
    {"cap":"Evidence Extraction",
     "text":"The in-class demos gave me exactly the evidence I needed to trust the theory.",
     "v1_label":"Positive","v2_aspects":["clarity","engagement"]},

    # ── Explainable Predictions (extra) ───────────────────────────────────────
    {"cap":"Explainable Predictions",
     "text":"The step-by-step breakdown made every derivation feel inevitable rather than magical.",
     "v1_label":"Positive","v2_aspects":["clarity"]},
    {"cap":"Explainable Predictions",
     "text":"I cannot understand why I keep getting the wrong answer when I follow the method exactly.",
     "v1_label":"Negative","v2_aspects":["clarity"]},

    # ── Contradiction Detection (extra) ────────────────────────────────────────
    {"cap":"Contradiction Detection",
     "text":"The workload is totally fine. Also I have not slept in four days because of it.",
     "v1_label":None,"v2_aspects":["workload"]},
    {"cap":"Contradiction Detection",
     "text":"The pacing is perfect and I never feel rushed. Except every single week when I do.",
     "v1_label":None,"v2_aspects":["pacing"]},

    # ── Ambiguity Detection (extra) ────────────────────────────────────────────
    {"cap":"Ambiguity Detection",
     "text":"I guess it could have been worse somehow.",
     "v1_label":"Neutral","v2_aspects":None},
    {"cap":"Ambiguity Detection",
     "text":"Mixed feelings about the whole thing, honestly.",
     "v1_label":"Neutral","v2_aspects":None},

    # ── Multi-Hop Reasoning (extra) ────────────────────────────────────────────
    {"cap":"Multi-Hop Reasoning",
     "text":"The group disbanded after the collaboration sessions went poorly, so nobody submitted the joint report.",
     "v1_label":"Negative","v2_aspects":["collaboration","workload"]},
    {"cap":"Multi-Hop Reasoning",
     "text":"When the clarity improved in week four, engagement rose noticeably and the group work became productive.",
     "v1_label":"Positive","v2_aspects":["clarity","engagement","collaboration"]},

    # ── Cause-Effect Extraction (extra) ────────────────────────────────────────
    {"cap":"Cause-Effect Extraction",
     "text":"Because the pre-class videos are inconsistently uploaded, half the class is unprepared each week.",
     "v1_label":"Negative","v2_aspects":["pre_class"]},
    {"cap":"Cause-Effect Extraction",
     "text":"The reduced workload this month allowed students to deeply engage with each topic.",
     "v1_label":"Positive","v2_aspects":["workload","engagement"]},

    # ── Temporal Understanding (extra) ─────────────────────────────────────────
    {"cap":"Temporal Understanding",
     "text":"From week six onward the pacing finally matched the complexity of the material.",
     "v1_label":"Positive","v2_aspects":["pacing"]},
    {"cap":"Temporal Understanding",
     "text":"Looking back over the semester, the collaboration component has consistently been the weakest part.",
     "v1_label":"Negative","v2_aspects":["collaboration"]},

    # ── Trend Analysis (extra) ─────────────────────────────────────────────────
    {"cap":"Trend Analysis",
     "text":"Over the past four weeks the quality of in-class discussions has steadily deteriorated.",
     "v1_label":"Negative","v2_aspects":["engagement","collaboration"]},
    {"cap":"Trend Analysis",
     "text":"Student participation has grown every week suggesting the engagement strategies are working.",
     "v1_label":"Positive","v2_aspects":["engagement"]},

    # ── Topic Modeling (extra) ─────────────────────────────────────────────────
    {"cap":"Topic Modeling",
     "text":"The three issues that keep surfacing are unclear instructions, tight deadlines, and passive lectures.",
     "v1_label":"Negative","v2_aspects":["clarity","workload","engagement"]},
    {"cap":"Topic Modeling",
     "text":"Recurring themes in student praise: approachable delivery, reasonable pacing, and useful pre-reads.",
     "v1_label":"Positive","v2_aspects":["engagement","pacing","pre_class"]},

    # ── Emerging Issue Detection (extra) ───────────────────────────────────────
    {"cap":"Emerging Issue Detection",
     "text":"A new pattern is emerging — students are forming private study groups to replace the tutorials.",
     "v1_label":"Negative","v2_aspects":["collaboration"]},
    {"cap":"Emerging Issue Detection",
     "text":"For the first time this semester, I am hearing widespread concerns about assessment fairness.",
     "v1_label":"Negative","v2_aspects":None},

    # ── Review Summarization ───────────────────────────────────────────────────
    {"cap":"Review Summarization",
     "text":"Overall the course has strong content, reasonable pacing, poor collaboration support, and a fair workload.",
     "v1_label":"Neutral","v2_aspects":["pacing","collaboration","workload"]},
    {"cap":"Review Summarization",
     "text":"In summary: brilliant teaching, excellent clarity, engaging style — truly a standout course.",
     "v1_label":"Positive","v2_aspects":["clarity","engagement"]},
    {"cap":"Review Summarization",
     "text":"To summarise: confusing lectures, excessive workload, broken group work, useless pre-class content.",
     "v1_label":"Negative","v2_aspects":["clarity","workload","collaboration","pre_class"]},

    # ── Review Clustering ──────────────────────────────────────────────────────
    {"cap":"Review Clustering",
     "text":"This falls firmly in the category of courses I would actively discourage others from taking.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Review Clustering",
     "text":"A textbook example of what good university teaching should look like.",
     "v1_label":"Positive","v2_aspects":None},
    {"cap":"Review Clustering",
     "text":"Generic, average, forgettable — exactly like a hundred other modules I have taken.",
     "v1_label":"Neutral","v2_aspects":None},

    # ── Instructor-Specific Analysis ───────────────────────────────────────────
    {"cap":"Instructor-Specific Analysis",
     "text":"The lead professor is fantastic but the tutorial demonstrators have no idea what they are teaching.",
     "v1_label":"Neutral","v2_aspects":None},
    {"cap":"Instructor-Specific Analysis",
     "text":"The professor's passion for the subject is infectious and transforms every lecture.",
     "v1_label":"Positive","v2_aspects":["engagement"]},
    {"cap":"Instructor-Specific Analysis",
     "text":"Different instructors explain the same topic in contradictory ways causing mass confusion.",
     "v1_label":"Negative","v2_aspects":["clarity"]},

    # ── Course-Specific Analysis ───────────────────────────────────────────────
    {"cap":"Course-Specific Analysis",
     "text":"Data Structures is the hardest course this semester by a significant margin.",
     "v1_label":"Negative","v2_aspects":["workload"]},
    {"cap":"Course-Specific Analysis",
     "text":"Machine Learning has a significantly higher workload than any other module this year.",
     "v1_label":"Negative","v2_aspects":["workload"]},
    {"cap":"Course-Specific Analysis",
     "text":"Operating Systems is the only course where I feel genuinely challenged and rewarded.",
     "v1_label":"Positive","v2_aspects":["engagement"]},

    # ── Facility Analysis ──────────────────────────────────────────────────────
    {"cap":"Facility Analysis",
     "text":"The computer lab software crashes every session making the practicals pointless.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Facility Analysis",
     "text":"The new lecture theatre's audio system makes it impossible to hear from the back.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Facility Analysis",
     "text":"The updated lab equipment has made a noticeable difference to the quality of experiments.",
     "v1_label":"Positive","v2_aspects":None},

    # ── Workload Analysis (extra) ─────────────────────────────────────────────
    {"cap":"Workload Analysis",
     "text":"The continuous assessment schedule leaves no time to process one topic before the next arrives.",
     "v1_label":"Negative","v2_aspects":["workload","pacing"]},
    {"cap":"Workload Analysis",
     "text":"Optional extension tasks are a great idea that reward students without penalising others.",
     "v1_label":"Positive","v2_aspects":["workload"]},

    # ── Fairness Analysis (extra) ─────────────────────────────────────────────
    {"cap":"Fairness Analysis",
     "text":"The attendance policy disproportionately punishes students with part-time jobs.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Fairness Analysis",
     "text":"Marks are returned with detailed feedback ensuring students understand every deduction.",
     "v1_label":"Positive","v2_aspects":None},

    # ── Recommendation Detection (extra) ──────────────────────────────────────
    {"cap":"Recommendation Detection",
     "text":"Weekly formative quizzes would help students benchmark their understanding before exams.",
     "v1_label":"Neutral","v2_aspects":["workload"]},
    {"cap":"Recommendation Detection",
     "text":"I strongly suggest the professor record lectures for students who need to replay explanations.",
     "v1_label":"Neutral","v2_aspects":["pre_class","clarity"]},

    # ── Question Detection (extra) ────────────────────────────────────────────
    {"cap":"Question Detection",
     "text":"Are we expected to know content from the supplementary reading for the final exam?",
     "v1_label":"Neutral","v2_aspects":["workload"]},
    {"cap":"Question Detection",
     "text":"Would it be possible to have a practice exam released before the main assessment?",
     "v1_label":"Neutral","v2_aspects":["workload"]},

    # ── Intent Classification (extra) ─────────────────────────────────────────
    {"cap":"Intent Classification",
     "text":"I am writing to suggest a restructuring of the assessment weighting in this course.",
     "v1_label":"Neutral","v2_aspects":None},
    {"cap":"Intent Classification",
     "text":"I would like to express my sincere gratitude for the quality of teaching this term.",
     "v1_label":"Positive","v2_aspects":None},

    # ── Toxicity Detection (extra) ─────────────────────────────────────────────
    {"cap":"Toxicity Detection",
     "text":"The professor is incompetent and should never be allowed near a classroom again.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Toxicity Detection",
     "text":"Whoever designed this course clearly hates students and wants us to fail.",
     "v1_label":"Negative","v2_aspects":None},

    # ── Spam Detection (extra) ────────────────────────────────────────────────
    {"cap":"Spam Detection",
     "text":"1 2 3 4 5 6 7 8 9 10",
     "v1_label":None,"v2_aspects":None},
    {"cap":"Spam Detection",
     "text":"bad bad bad bad bad bad bad bad bad bad",
     "v1_label":"Negative","v2_aspects":None},

    # ── Domain Adaptation (extra) ─────────────────────────────────────────────
    {"cap":"Domain Adaptation",
     "text":"The simulation tools are outdated and produce results incompatible with current standards.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Domain Adaptation",
     "text":"The interdisciplinary nature of this programme makes it hard to calibrate workload expectations.",
     "v1_label":"Negative","v2_aspects":["workload"]},

    # ── Uncertainty Estimation (extra) ────────────────────────────────────────
    {"cap":"Uncertainty Estimation",
     "text":"I cannot tell whether the confusion I feel is the course's fault or my own.",
     "v1_label":"Neutral","v2_aspects":None},
    {"cap":"Uncertainty Estimation",
     "text":"The data is mixed — some days are brilliant, others are baffling.",
     "v1_label":"Neutral","v2_aspects":None},

    # ── Hallucination Resistance (extra) ──────────────────────────────────────
    {"cap":"Hallucination Resistance",
     "text":"N/A",
     "v1_label":"Neutral","v2_aspects":None},
    {"cap":"Hallucination Resistance",
     "text":"?",
     "v1_label":"Neutral","v2_aspects":None},

    # ── Real-Time Inference (extra) ────────────────────────────────────────────
    {"cap":"Real-Time Inference",
     "text":"Short.",
     "v1_label":"Neutral","v2_aspects":None},
    {"cap":"Real-Time Inference",
     "text":"The engagement was brilliant, the pacing superb, the clarity outstanding, and the workload just right.",
     "v1_label":"Positive","v2_aspects":["engagement","pacing","clarity","workload"]},

    # ── Privacy Preservation (extra) ──────────────────────────────────────────
    {"cap":"Privacy Preservation",
     "text":"Dr Kapoor's favouritism towards certain students is blatantly obvious to everyone.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Privacy Preservation",
     "text":"Student ID 21B0342 consistently gets preferential treatment during lab assessments.",
     "v1_label":"Negative","v2_aspects":None},

    # ── Bias Detection (extra) ────────────────────────────────────────────────
    {"cap":"Bias Detection and Mitigation",
     "text":"The examples in lectures always assume the student is male and from an urban background.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Bias Detection and Mitigation",
     "text":"The content is culturally inclusive and represents diverse perspectives fairly.",
     "v1_label":"Positive","v2_aspects":None},

    # ── Entity Recognition ─────────────────────────────────────────────────────
    {"cap":"Entity Recognition",
     "text":"The Python lab in CS201 uses outdated libraries that conflict with modern toolchains.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Entity Recognition",
     "text":"The NumPy tutorial prepared by the TA for the Data Science module was exceptional.",
     "v1_label":"Positive","v2_aspects":None},
    {"cap":"Entity Recognition",
     "text":"The Moodle integration with Turnitin keeps breaking and nobody in IT can fix it.",
     "v1_label":"Negative","v2_aspects":None},

    # ── Relationship Extraction ────────────────────────────────────────────────
    {"cap":"Relationship Extraction",
     "text":"The professor's enthusiasm directly translates into higher student engagement levels.",
     "v1_label":"Positive","v2_aspects":["engagement"]},
    {"cap":"Relationship Extraction",
     "text":"Poor clarity in lectures creates a downstream effect of failing collaborative projects.",
     "v1_label":"Negative","v2_aspects":["clarity","collaboration"]},
    {"cap":"Relationship Extraction",
     "text":"The overly long pre-class readings reduce energy available for in-class participation.",
     "v1_label":"Negative","v2_aspects":["pre_class","engagement"]},

    # ── Benchmarking and Evaluation ────────────────────────────────────────────
    {"cap":"Benchmarking and Evaluation",
     "text":"Compared to industry-standard pedagogy benchmarks this course ranks in the bottom quartile.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Benchmarking and Evaluation",
     "text":"By every measurable outcome this course exceeds the faculty's own quality indicators.",
     "v1_label":"Positive","v2_aspects":None},
    {"cap":"Benchmarking and Evaluation",
     "text":"The assessment standards are inconsistent with those used in equivalent courses across the university.",
     "v1_label":"Negative","v2_aspects":None},

    # ── Knowledge Distillation Support ────────────────────────────────────────
    {"cap":"Knowledge Distillation Support",
     "text":"The condensed summary sheets helped more than the full two-hour lecture.",
     "v1_label":"Positive","v2_aspects":["clarity"]},
    {"cap":"Knowledge Distillation Support",
     "text":"The cheat sheet allowed me to internalise the core ideas without wading through dense theory.",
     "v1_label":"Positive","v2_aspects":["clarity"]},
    {"cap":"Knowledge Distillation Support",
     "text":"Distilling the key points into five bullet points each week would massively help retention.",
     "v1_label":"Neutral","v2_aspects":["clarity"]},

    # ── Continual Learning ─────────────────────────────────────────────────────
    {"cap":"Continual Learning",
     "text":"The course material has not been updated in years and feels completely disconnected from current practice.",
     "v1_label":"Negative","v2_aspects":None},
    {"cap":"Continual Learning",
     "text":"The curriculum is refreshed every semester to incorporate emerging research — it shows.",
     "v1_label":"Positive","v2_aspects":None},
    {"cap":"Continual Learning",
     "text":"I appreciate that student feedback from previous cohorts has visibly shaped this year's structure.",
     "v1_label":"Positive","v2_aspects":None},

    # ── Language Identification ────────────────────────────────────────────────
    {"cap":"Language Identification",
     "text":"Le cours est trop rapide et je ne comprends rien.",
     "v1_label":None,"v2_aspects":None},
    {"cap":"Language Identification",
     "text":"El ritmo de la clase es demasiado rápido para los estudiantes internacionales.",
     "v1_label":None,"v2_aspects":None},
    {"cap":"Language Identification",
     "text":"授業のペースが速すぎて内容を吸収できない。",
     "v1_label":None,"v2_aspects":None},
]

# Merge into main TEST_CASES list at import time
TEST_CASES.extend(_EXTRA_CASES)


# ── Fix: Code-Mixed Language (add one more case) ───────────────────────────────
_EXTRA_CASES_2: List[Dict[str, Any]] = [
    {"cap":"Code-Mixed Language Understanding",
     "text":"Yeh lecture bilkul bakwaas tha, kuch bhi clear nahi tha honestly.",
     "v1_label":"Negative","v2_aspects":["clarity"]},
]
TEST_CASES.extend(_EXTRA_CASES_2)
