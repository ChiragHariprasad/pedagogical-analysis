"""
Comprehensive test suite for the ABSA pipeline.

Run with:
    python test_pipeline.py

Exercises:
    * Pure English positive / negative feedback
    * Hinglish code-mixed inputs
    * Contradictory sentiments in a single sentence
    * Multiple aspects in one sentence
    * All 6 pedagogy types referenced
    * Edge cases: empty, single word, very long text
"""

from __future__ import annotations

import textwrap
import time
from typing import Any, Dict, List

from nlp_pipeline import run_absa_pipeline

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES: List[Dict[str, str]] = [
    # 1 – Pure English positive
    {
        "label": "English positive",
        "text": "The lecture was excellent and the professor explained every concept clearly.",
    },
    # 2 – Pure English negative
    {
        "label": "English negative",
        "text": "The grading was unfair and the exam questions were confusing and irrelevant.",
    },
    # 3 – Hinglish code-mixed
    {
        "label": "Hinglish mixed",
        "text": "PBL bahut achha tha but lectures bekar the.",
    },
    # 4 – Contradictory sentiments
    {
        "label": "Contradictory",
        "text": "The project was amazing but grading was terrible.",
    },
    # 5 – Multiple aspects in one sentence
    {
        "label": "Multiple aspects",
        "text": "The lab equipment is outdated, the WiFi keeps dropping, and the projector barely works.",
    },
    # 6 – Traditional Lecture
    {
        "label": "Traditional Lecture",
        "text": "Traditional lecture with PowerPoint slides made the session very boring and monotonous.",
    },
    # 7 – PBL
    {
        "label": "Project-Based Learning",
        "text": "The project-based approach was hands-on and practical, we learned real coding skills.",
    },
    # 8 – Flipped Classroom
    {
        "label": "Flipped Classroom",
        "text": "Watching videos before class was useful, the classroom discussion was interactive and engaging.",
    },
    # 9 – Collaborative
    {
        "label": "Collaborative Learning",
        "text": "Pair programming and group work helped me understand difficult concepts through peer explanation.",
    },
    # 10 – Inquiry-Based
    {
        "label": "Inquiry-Based Learning",
        "text": "The inquiry exercises made us think deeply, although some questions felt too vague.",
    },
    # 11 – Experiential Labs
    {
        "label": "Experiential Labs",
        "text": "The hands-on lab was zabardast, coding exercises were mazedar and the demo was lajawab.",
    },
    # 12 – Edge: empty
    {
        "label": "Edge: empty",
        "text": "",
    },
    # 13 – Edge: single word
    {
        "label": "Edge: single word",
        "text": "Good",
    },
    # 14 – Edge: very long text
    {
        "label": "Edge: long text (500+ chars)",
        "text": (
            "The entire semester was a mixed bag. The project-based learning component was "
            "absolutely fantastic — we built a real application from scratch and the professor "
            "guided us through every step. However, the traditional lectures were extremely "
            "boring and the slides were just walls of text. The lab sessions had outdated "
            "equipment and the WiFi kept disconnecting. On the positive side, the flipped "
            "classroom approach with pre-recorded videos was very helpful for revision. "
            "The grading rubric was unclear and marks seemed arbitrary. Collaborative group "
            "work was okay but some team members did not contribute. Overall, the syllabus "
            "needs a major update to include modern topics and the teaching methodology should "
            "shift towards more hands-on and interactive methods."
        ),
    },
    # 15 – Hinglish with negation
    {
        "label": "Hinglish negation",
        "text": "Professor ki teaching bilkul samajh nahi aayi, bahut mushkil tha.",
    },
]


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------


def _print_separator(char: str = "═", width: int = 90) -> None:
    print(char * width)


def _print_result(idx: int, case: Dict[str, str], result: Dict[str, Any]) -> None:
    _print_separator()
    print(f"  Test {idx:>2}: {case['label']}")
    _print_separator("─")
    print(f"  Original  : {result['original_feedback'][:100]}")
    print(f"  Processed : {result['processed_feedback'][:100]}")
    print(f"  Language   : {result['language']}")
    print(f"  Aspects    : {len(result['aspects'])}")
    if result["aspects"]:
        print()
        header = f"  {'Aspect':<18} {'Category':<16} {'Sentiment':<10} {'Score':>7}  Context"
        print(header)
        print("  " + "─" * 86)
        for a in result["aspects"]:
            ctx = textwrap.shorten(a["context_clause"], width=35, placeholder="…")
            print(
                f"  {a['aspect']:<18} {a['category']:<16} "
                f"{a['sentiment']:<10} {a['polarity_score']:>7.4f}  {ctx}"
            )
    else:
        print("  (no aspects extracted)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print()
    _print_separator("▓")
    print("  PEDAGOGICAL INTELLIGENCE SYSTEM – ABSA PIPELINE TEST SUITE")
    _print_separator("▓")
    print()

    total_start = time.perf_counter()

    for idx, case in enumerate(TEST_CASES, start=1):
        t0 = time.perf_counter()
        result = run_absa_pipeline(case["text"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _print_result(idx, case, result)
        print(f"\n  ⏱  {elapsed_ms:.0f} ms")
        print()

    total_elapsed = time.perf_counter() - total_start
    _print_separator("▓")
    print(f"  All {len(TEST_CASES)} tests completed in {total_elapsed:.1f}s")
    _print_separator("▓")
    print()


if __name__ == "__main__":
    main()
