"""
Curated Pedagogical Aspect Lexicon for the Pedagogical Intelligence System.

Provides structured vocabulary for aspect-based sentiment analysis (ABSA) in
educational feedback. Covers six pedagogical categories and includes a
comprehensive Hinglish→English transliteration dictionary tailored to
Indian engineering-college contexts (RV College of Engineering, Bengaluru).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Set

# ---------------------------------------------------------------------------
# Aspect categories – each maps to a frozenset of lowercase trigger words.
# ---------------------------------------------------------------------------

ASPECT_CATEGORIES: Dict[str, FrozenSet[str]] = {
    "TEACHING": frozenset({
        "lecture", "professor", "teacher", "instructor", "explanation",
        "teaching", "faculty", "sir", "madam", "class", "session",
    }),
    "ASSESSMENT": frozenset({
        "grading", "exam", "marks", "evaluation", "assignment", "quiz",
        "test", "scoring", "rubric", "feedback", "grade",
    }),
    "INFRASTRUCTURE": frozenset({
        "lab", "library", "projector", "seating", "ac", "wifi",
        "classroom", "equipment", "computer", "internet",
    }),
    "CURRICULUM": frozenset({
        "syllabus", "course", "content", "material", "textbook",
        "topics", "module", "chapter", "subject", "theory",
    }),
    "METHODOLOGY": frozenset({
        "project", "hands-on", "group", "presentation", "coding",
        "practical", "activity", "exercise", "demo", "workshop",
        "pair", "team",
    }),
    "EXPERIENCE": frozenset({
        "engagement", "interest", "boring", "fun", "interactive",
        "useful", "helpful", "difficult", "easy", "challenging",
        "motivation", "understanding",
    }),
}

# ---------------------------------------------------------------------------
# Flat set that unions every category – used for O(1) membership checks.
# ---------------------------------------------------------------------------

ALL_ASPECTS: FrozenSet[str] = frozenset().union(*ASPECT_CATEGORIES.values())

# ---------------------------------------------------------------------------
# Reverse lookup cache (built once at import time).
# ---------------------------------------------------------------------------

_WORD_TO_CATEGORY: Dict[str, str] = {}
for _cat, _words in ASPECT_CATEGORIES.items():
    for _w in _words:
        _WORD_TO_CATEGORY[_w] = _cat


def get_aspect_category(word: str) -> Optional[str]:
    """Return the category name for *word*, or ``None`` if not found.

    Lookup is case-insensitive and runs in O(1).

    >>> get_aspect_category("lecture")
    'TEACHING'
    >>> get_aspect_category("xyz") is None
    True
    """
    return _WORD_TO_CATEGORY.get(word.lower())


# ---------------------------------------------------------------------------
# Hinglish / Hindi-transliterated → English dictionary  (80+ entries)
# ---------------------------------------------------------------------------

HINGLISH_DICT: Dict[str, str] = {
    # -- Quality / sentiment words ------------------------------------------
    "achha": "good",
    "acha": "good",
    "accha": "good",
    "bekar": "bad",
    "bakwas": "terrible",
    "mast": "great",
    "badhiya": "excellent",
    "badiya": "excellent",
    "ghatiya": "poor",
    "zabardast": "fantastic",
    "kamaal": "wonderful",
    "kamal": "wonderful",
    "shandaar": "magnificent",
    "shandar": "magnificent",
    "lajawab": "outstanding",
    "behtareen": "best",
    "behtarin": "best",
    "wahiyat": "awful",
    "ganda": "bad",
    "sahi": "correct",
    "galat": "wrong",
    "thik": "okay",
    "theek": "okay",
    "mazedaar": "enjoyable",
    "mazedar": "enjoyable",
    "jabardast": "fantastic",
    "shandarr": "magnificent",
    "acchi": "good",
    "bura": "bad",
    "buri": "bad",
    "sasta": "cheap",

    # -- Degree / intensifier words -----------------------------------------
    "bahut": "very",
    "bhot": "very",
    "bohot": "very",
    "boht": "very",
    "thoda": "little",
    "thodi": "little",
    "zyada": "more",
    "jyada": "more",
    "kam": "less",
    "bilkul": "absolutely",
    "ekdum": "totally",
    "poora": "complete",
    "pura": "complete",
    "kaafi": "quite",
    "itna": "this much",
    "utna": "that much",
    "sabse": "most",

    # -- Common verbs / connectors ------------------------------------------
    "samajh": "understand",
    "samjh": "understand",
    "nahi": "not",
    "nahin": "not",
    "nhi": "not",
    "bhi": "also",
    "sirf": "only",
    "kuch": "some",
    "lekin": "but",
    "magar": "but",
    "par": "but",
    "aur": "and",
    "tha": "was",
    "thi": "was",
    "the": "were",
    "hai": "is",
    "hain": "are",
    "wala": "one",
    "wali": "one",
    "koi": "any",
    "kya": "what",
    "kyun": "why",
    "kaise": "how",
    "abhi": "now",
    "pehle": "before",
    "baad": "after",

    # -- Education / academic terms -----------------------------------------
    "padhai": "studies",
    "padhana": "teach",
    "padhao": "teach",
    "likhna": "write",
    "seekhna": "learn",
    "sikhana": "teach",
    "sikhna": "learn",
    "mushkil": "difficult",
    "aasan": "easy",
    "asan": "easy",
    "mehnat": "hard work",
    "tareeka": "method",
    "tarika": "method",
    "jawab": "answer",
    "sawal": "question",
    "kitab": "book",
    "kaksha": "class",
    "pariksha": "exam",
    "pathyakram": "syllabus",
}
