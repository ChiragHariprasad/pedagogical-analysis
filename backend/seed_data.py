"""
Seed script for the Pedagogical Intelligence System.

Generates 30 realistic sample survey responses covering all 6 pedagogy types,
runs the ABSA pipeline on each, and inserts them into the SQLite database.

Distribution goals:
    * All 6 pedagogies represented (5 surveys × 6 pedagogies = 30 responses)
    * ~30 % Hinglish feedback
    * ~20 % contradictory sentiments
    * Ratings span 1–5 with a realistic bell-curve-ish distribution

Usage:
    python seed_data.py
"""

from __future__ import annotations

import random
import sys
import uuid
from datetime import datetime, timezone

from database import init_db, insert_survey, wipe_all_data
from models import PEDAGOGIES, pedagogy_name
from nlp_pipeline import run_absa_pipeline

# ---------------------------------------------------------------------------
# Seed feedback pool – keyed by pedagogy_id
# ---------------------------------------------------------------------------

FEEDBACK_POOL: dict[str, list[str]] = {
    "traditional_lecture": [
        "The lecture was clear and the professor covered every topic in detail.",
        "Slides were just walls of text, the session felt extremely monotonous and boring.",
        "Sir ki teaching bahut achhi thi, sab samajh aa gaya.",
        "Lecture content was okay but the pace was too fast for the difficult topics.",
        # contradictory
        "The explanation was excellent but the class session was too long and tiring.",
    ],
    "project_based": [
        "The project was fantastic — we built a real-world application from scratch!",
        "Project bahut mast tha, coding seekhne mein bahut help mili.",
        "The project deadlines were unrealistic and the grading rubric was unclear.",
        "Hands-on coding exercise was very useful and engaging for the whole team.",
        # contradictory
        "Project approach was zabardast but the evaluation was ghatiya and unfair.",
    ],
    "flipped_classroom": [
        "Watching videos before class was really helpful, the discussion was interactive.",
        "The pre-class material was too long and boring, nobody completed it.",
        "Flipped classroom achha concept hai but execution theek nahi thi.",
        "I loved the in-class activities after watching the lecture videos at home.",
        # contradictory
        "The content was badhiya but the classroom discussion felt rushed and shallow.",
    ],
    "collaborative": [
        "Pair programming helped me understand difficult concepts through peer explanation.",
        "Group work was terrible — some team members did not contribute at all.",
        "Collaborative learning bahut engaging tha, team ke saath kaam karna mazedar tha.",
        "The peer code review exercise improved my coding skills significantly.",
        # contradictory
        "Group activity was fun but the presentation grading was confusing and strict.",
    ],
    "inquiry_based": [
        "The inquiry exercises made us think deeply and explore creative solutions.",
        "Questions were too vague and there was no guidance, very frustrating experience.",
        "Problem-based learning se bahut interest aaya, mushkil par challenging tha.",
        "Investigating real problems was engaging and improved my critical thinking.",
        # contradictory
        "The inquiry method was interesting but the assessment was bekar and irrelevant.",
    ],
    "experiential_labs": [
        "The hands-on lab was amazing, we got to experiment with real equipment.",
        "Lab equipment was outdated and the WiFi kept disconnecting during the session.",
        "Practical coding exercises were zabardast, demo was lajawab aur bahut helpful.",
        "The lab exercise was the best part of the course, very useful for understanding.",
        # contradictory
        "Lab activity was excellent but the classroom seating was uncomfortable and AC broken.",
    ],
}

# Rating presets – (effectiveness, engagement, clarity, relevance)
# Each tuple is designed for a realistic spread
RATING_PRESETS: list[tuple[int, int, int, int]] = [
    (5, 5, 4, 5),  # very positive
    (4, 4, 4, 4),  # positive
    (4, 3, 4, 3),  # mixed-positive
    (3, 3, 3, 3),  # neutral
    (3, 2, 3, 2),  # mixed-negative
    (2, 2, 2, 3),  # negative
    (2, 1, 2, 1),  # very negative
    (1, 1, 1, 2),  # terrible
    (5, 4, 5, 4),  # strong positive
    (3, 4, 2, 3),  # contradictory feel
]


def _pick_ratings() -> tuple[int, int, int, int]:
    """Return a random but plausible rating tuple."""
    base = random.choice(RATING_PRESETS)
    # Add small jitter (±1 clamped to 1-5)
    return tuple(  # type: ignore[return-value]
        max(1, min(5, v + random.choice([-1, 0, 0, 0, 1]))) for v in base
    )


def generate_seed_data() -> list[dict]:
    """Build 30 response dicts ready for insertion."""
    responses: list[dict] = []
    for ped in PEDAGOGIES:
        pid = ped["id"]
        pool = FEEDBACK_POOL[pid]
        for fb_text in pool:
            eff, eng, cla, rel = _pick_ratings()
            responses.append(
                {
                    "pedagogy_id": pid,
                    "pedagogy_name": pedagogy_name(pid),
                    "effectiveness": eff,
                    "engagement": eng,
                    "clarity": cla,
                    "relevance": rel,
                    "feedback": fb_text,
                }
            )
    return responses


def main() -> None:
    print("=" * 70)
    print("  PEDAGOGICAL INTELLIGENCE SYSTEM - SEED DATA GENERATOR (v3)")
    print("=" * 70)
    print()

    # Initialise DB and wipe old data
    init_db()
    print("  Wiping existing data ...")
    wipe_all_data()
    print("  [OK] Database wiped.\n")

    raw_responses = generate_seed_data()
    random.shuffle(raw_responses)

    print(f"  Generated {len(raw_responses)} raw responses.")
    print("  Running ABSA pipeline on each (this may take a minute) ...\n")

    # Group into surveys of 6 (one per pedagogy) to mimic realistic submissions
    surveys: list[list[dict]] = []
    batch: list[dict] = []
    for r in raw_responses:
        batch.append(r)
        if len(batch) == 6:
            surveys.append(batch)
            batch = []
    if batch:
        surveys.append(batch)

    total_inserted = 0
    for i, survey_batch in enumerate(surveys, start=1):
        survey_id = uuid.uuid4().hex[:12]
        submitted_at = datetime.now(timezone.utc).isoformat()

        enriched: list[dict] = []
        for r in survey_batch:
            print(f"    [{total_inserted + 1:>2}/30] Analysing: {r['feedback'][:60]}…")
            absa = run_absa_pipeline(r["feedback"])
            r["absa_result"] = absa
            enriched.append(r)
            total_inserted += 1

        insert_survey(survey_id, submitted_at, enriched, student_email=None)
        print(f"  [OK] Survey {i} ({survey_id}) inserted with {len(enriched)} responses.\n")

    print("=" * 70)
    print(f"  Done! Inserted {total_inserted} responses across {len(surveys)} surveys.")
    print("=" * 70)


if __name__ == "__main__":
    main()
