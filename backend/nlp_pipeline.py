"""
Aspect-Based Sentiment Analysis (ABSA) pipeline for the Pedagogical
Intelligence System.

Architecture
------------
1. **preprocess_hinglish** – detect language via *langdetect*; replace
   Hinglish tokens using the curated ``HINGLISH_DICT``.
2. **split_clauses** – break compound sentences at conjunctions so each
   clause can be scored independently (handles contradictory sentiments).
3. **extract_aspects** – SpaCy dependency parsing to find NOUNs/PROPNs in
   the aspect lexicon, then gather each aspect's governing-verb subtree as
   context and collect descriptors (``amod`` / ``advmod``).
4. **classify_sentiment** – run a HuggingFace transformer on the clause to
   produce ``{sentiment, polarity_score}``.
5. **run_absa_pipeline** – orchestrates all steps and returns an
   ``ABSAResult``-shaped dict.

Model selection (tried in order):
    * ``yangheng/deberta-v3-base-absa-v1.1``
    * ``nlptown/bert-base-multilingual-uncased-sentiment`` (5-star)
    * ``distilbert-base-uncased-finetuned-sst-2-english``
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import spacy
from langdetect import detect, LangDetectException

from aspect_lexicon import ALL_ASPECTS, HINGLISH_DICT, get_aspect_category

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (avoids heavy imports at module level)
# ---------------------------------------------------------------------------

_nlp: Optional[spacy.language.Language] = None
_sentiment_pipe: Optional[Any] = None
_sentiment_model_name: Optional[str] = None


def _get_spacy() -> spacy.language.Language:
    """Load the SpaCy English model once, downloading if necessary."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("SpaCy model not found - downloading en_core_web_sm...")
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_sentiment_pipeline() -> Any:
    """Load a HuggingFace sentiment-analysis pipeline with cascading fallback.

    Tried in order:
    1. ``yangheng/deberta-v3-base-absa-v1.1``
    2. ``nlptown/bert-base-multilingual-uncased-sentiment``
    3. ``distilbert-base-uncased-finetuned-sst-2-english``
    """
    global _sentiment_pipe, _sentiment_model_name
    if _sentiment_pipe is not None:
        return _sentiment_pipe

    from transformers import pipeline as hf_pipeline  # heavy – imported lazily

    candidates = [
        "yangheng/deberta-v3-base-absa-v1.1",
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "distilbert-base-uncased-finetuned-sst-2-english",
    ]
    for model_name in candidates:
        try:
            logger.info("Attempting to load sentiment model: %s", model_name)
            _sentiment_pipe = hf_pipeline(
                "sentiment-analysis", model=model_name, truncation=True
            )
            _sentiment_model_name = model_name
            logger.info("Successfully loaded: %s", model_name)
            return _sentiment_pipe
        except Exception as exc:
            logger.warning(
                "Failed to load %s: %s – trying next fallback.", model_name, exc
            )
    raise RuntimeError("Could not load any sentiment model.")


# ---------------------------------------------------------------------------
# Conjunction / clause-split pattern
# ---------------------------------------------------------------------------

_CLAUSE_SPLIT_RE = re.compile(
    r"\s*\b(?:but|however|although|yet|while|whereas|nevertheless|"
    r"still|though|lekin|magar|par)\b\s*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public pipeline functions
# ---------------------------------------------------------------------------


def preprocess_hinglish(text: str) -> Tuple[str, str]:
    """Detect language and apply Hinglish→English word replacement.

    Returns
    -------
    (language_code, processed_text)
    """
    text = text.strip()
    if not text:
        return ("unknown", "")

    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"

    words = text.split()
    processed: List[str] = []
    for word in words:
        # Strip punctuation for lookup, but keep original if no match
        clean = re.sub(r"[^\w]", "", word.lower())
        if clean in HINGLISH_DICT:
            processed.append(HINGLISH_DICT[clean])
        else:
            processed.append(word)

    return lang, " ".join(processed)


def split_clauses(text: str) -> List[str]:
    """Split *text* at adversative conjunctions into independent clauses.

    If the text does not contain any splitting trigger, returns a
    single-element list with the original text.
    """
    if not text.strip():
        return []
    parts = _CLAUSE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def extract_aspects(text: str) -> List[Dict[str, Any]]:
    """Use SpaCy dependency parsing to find pedagogical aspects.

    For each NOUN/PROPN that matches the aspect lexicon:
    - determines its category
    - collects the governing-verb's subtree as *context_clause*
    - collects ``amod`` / ``advmod`` children as *descriptors*

    Returns a list of dicts: ``{aspect, category, context_clause, descriptors}``.
    """
    nlp = _get_spacy()
    doc = nlp(text)

    seen_aspects: set[str] = set()
    results: List[Dict[str, Any]] = []

    for token in doc:
        if token.pos_ not in ("NOUN", "PROPN"):
            continue

        word_lower = token.text.lower()

        # Check both the raw word and its lemma against the lexicon
        matched_word: Optional[str] = None
        if word_lower in ALL_ASPECTS:
            matched_word = word_lower
        elif token.lemma_.lower() in ALL_ASPECTS:
            matched_word = token.lemma_.lower()
        else:
            continue

        # Avoid duplicate aspects within the same analysis run
        if matched_word in seen_aspects:
            continue
        seen_aspects.add(matched_word)

        category = get_aspect_category(matched_word) or "EXPERIENCE"

        # Collect descriptors (adjective / adverb modifiers)
        descriptors: List[str] = [
            child.text
            for child in token.children
            if child.dep_ in ("amod", "advmod")
        ]

        # Also grab descriptors attached to the head verb (complement etc.)
        head = token.head
        if head.pos_ in ("VERB", "AUX"):
            for child in head.children:
                if child.dep_ in ("acomp", "attr", "advmod") and child != token:
                    descriptors.append(child.text)

        # Build context clause from the governing verb's subtree
        context_clause = _build_context(token, doc)

        results.append(
            {
                "aspect": matched_word,
                "category": category,
                "context_clause": context_clause,
                "descriptors": descriptors,
            }
        )

    return results


def _build_context(token: spacy.tokens.Token, doc: spacy.tokens.Doc) -> str:
    """Return a clause-level context string for *token*."""
    # Walk up to the nearest VERB / AUX head
    head = token.head
    depth = 0
    while head.pos_ not in ("VERB", "AUX") and head != head.head and depth < 5:
        head = head.head
        depth += 1

    if head.pos_ in ("VERB", "AUX"):
        subtree_tokens = sorted(head.subtree, key=lambda t: t.i)
        return " ".join(t.text for t in subtree_tokens)

    # Fallback: window of ±5 tokens around the aspect
    start = max(0, token.i - 5)
    end = min(len(doc), token.i + 6)
    return doc[start:end].text


def classify_sentiment(clause: str) -> Dict[str, Any]:
    """Run the transformer model on *clause* and return normalised sentiment.

    Returns ``{sentiment: str, polarity_score: float}`` where sentiment is one
    of ``Positive``, ``Negative``, ``Neutral`` and polarity_score ∈ [-1, 1].
    """
    if not clause.strip():
        return {"sentiment": "Neutral", "polarity_score": 0.0}

    pipe = _get_sentiment_pipeline()
    result = pipe(clause[:512])[0]  # truncate to model max
    label: str = result["label"]
    score: float = result["score"]

    global _sentiment_model_name

    if _sentiment_model_name and "nlptown" in _sentiment_model_name:
        # 5-star model: labels are "1 star" .. "5 stars"
        stars = int(label.split()[0])
        if stars >= 4:
            return {"sentiment": "Positive", "polarity_score": round((stars - 3) / 2, 4)}
        elif stars <= 2:
            return {"sentiment": "Negative", "polarity_score": round((stars - 3) / 2, 4)}
        else:
            return {"sentiment": "Neutral", "polarity_score": 0.0}
    else:
        # Binary model (POSITIVE / NEGATIVE)
        if label.upper() == "POSITIVE":
            polarity = round(score, 4)
            sentiment = "Positive" if score >= 0.6 else "Neutral"
        else:
            polarity = round(-score, 4)
            sentiment = "Negative" if score >= 0.6 else "Neutral"
        return {"sentiment": sentiment, "polarity_score": polarity}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_absa_pipeline(feedback: str) -> Dict[str, Any]:
    """Run the full ABSA pipeline on *feedback* and return a dict matching
    the ``ABSAResult`` schema.

    Handles:
    * Hinglish preprocessing
    * Clause splitting for contradictory sentiments
    * Aspect extraction with category tagging
    * Per-clause sentiment classification
    * Empty / minimal input (returns empty aspects list)
    """
    feedback = (feedback or "").strip()
    if not feedback:
        return {
            "original_feedback": feedback,
            "processed_feedback": "",
            "language": "unknown",
            "aspects": [],
        }

    # Step 1 – Hinglish preprocessing
    lang, processed = preprocess_hinglish(feedback)

    # Step 2 – Clause splitting
    clauses = split_clauses(processed)

    # Step 3 & 4 – Extract aspects and classify sentiment per clause
    all_aspects: List[Dict[str, Any]] = []
    seen_globally: set[str] = set()

    for clause in clauses:
        aspect_dicts = extract_aspects(clause)
        for a in aspect_dicts:
            if a["aspect"] in seen_globally:
                continue
            seen_globally.add(a["aspect"])

            sent = classify_sentiment(a["context_clause"])
            all_aspects.append(
                {
                    "aspect": a["aspect"],
                    "category": a["category"],
                    "sentiment": sent["sentiment"],
                    "polarity_score": sent["polarity_score"],
                    "context_clause": a["context_clause"],
                    "descriptors": a["descriptors"],
                }
            )

    # If no lexicon-matched aspects were found, try the whole text as a
    # single "general" aspect so the caller always gets some signal.
    if not all_aspects:
        sent = classify_sentiment(processed)
        all_aspects.append(
            {
                "aspect": "general",
                "category": "EXPERIENCE",
                "sentiment": sent["sentiment"],
                "polarity_score": sent["polarity_score"],
                "context_clause": processed,
                "descriptors": [],
            }
        )

    return {
        "original_feedback": feedback,
        "processed_feedback": processed,
        "language": lang,
        "aspects": all_aspects,
    }
