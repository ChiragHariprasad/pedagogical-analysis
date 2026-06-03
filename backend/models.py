"""
Pydantic models for the Pedagogical Intelligence System.

Defines request/response schemas for the survey API, ABSA pipeline output,
analytics, and the canonical list of supported pedagogy types.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Canonical pedagogy types (used by the survey form & analytics)
# ---------------------------------------------------------------------------

PEDAGOGIES: List[dict] = [
    {
        "id": "traditional_lecture",
        "name": "Traditional Lecture",
        "description": (
            "Instructor-led direct instruction with "
            "PowerPoint/whiteboard presentations"
        ),
    },
    {
        "id": "project_based",
        "name": "Project-Based Learning (PBL)",
        "description": (
            "Deep, hands-on exploration of real-world problems "
            "over extended periods"
        ),
    },
    {
        "id": "flipped_classroom",
        "name": "Flipped Classroom",
        "description": (
            "Pre-class materials (videos/readings), class time "
            "for active application & discussion"
        ),
    },
    {
        "id": "collaborative",
        "name": "Collaborative / Peer Learning",
        "description": (
            "Group work, pair programming, peer code reviews, "
            "shared responsibility"
        ),
    },
    {
        "id": "inquiry_based",
        "name": "Inquiry / Problem-Based Learning",
        "description": (
            "Learning driven by open-ended questions and problems "
            "requiring investigation"
        ),
    },
    {
        "id": "experiential_labs",
        "name": "Experiential / Hands-On Labs",
        "description": (
            "Direct experience through labs, practical coding "
            "exercises, and demos"
        ),
    },
]

# Quick lookup helpers
_PEDAGOGY_MAP = {p["id"]: p for p in PEDAGOGIES}


def pedagogy_name(pedagogy_id: str) -> str:
    """Return the display name for *pedagogy_id*, or the raw id as fallback."""
    entry = _PEDAGOGY_MAP.get(pedagogy_id)
    return entry["name"] if entry else pedagogy_id


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class PedagogyRating(BaseModel):
    """A student's anonymous rating of one pedagogy type."""

    pedagogy_id: str
    effectiveness: int = Field(
        ..., ge=0, le=5, description="How effective was this method"
    )
    engagement: int = Field(
        ..., ge=0, le=5, description="How engaging was this method"
    )
    clarity: int = Field(
        ..., ge=0, le=5, description="How clear was the instruction"
    )
    relevance: int = Field(
        ..., ge=0, le=5, description="How relevant to learning goals"
    )
    feedback: str = Field(
        ..., min_length=1, description="Qualitative feedback text"
    )


class SurveySubmission(BaseModel):
    """One anonymous survey submission containing ratings for ≥1 pedagogies."""

    responses: List[PedagogyRating]
    submitted_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# ABSA result models
# ---------------------------------------------------------------------------


class AspectSentiment(BaseModel):
    """Single aspect extracted from feedback with its sentiment."""

    aspect: str
    category: str  # TEACHING, ASSESSMENT, …
    sentiment: str  # Positive, Negative, Neutral
    polarity_score: float
    context_clause: str
    descriptors: List[str] = []


class ABSAResult(BaseModel):
    """Full ABSA pipeline output for one feedback string."""

    original_feedback: str
    processed_feedback: str
    language: str
    aspects: List[AspectSentiment]


# ---------------------------------------------------------------------------
# Response / read models
# ---------------------------------------------------------------------------


class ResponseRecord(BaseModel):
    """A single stored response, including ABSA analysis."""

    id: int
    survey_id: str
    pedagogy_id: str
    pedagogy_name: str
    effectiveness: int
    engagement: int
    clarity: int
    relevance: int
    feedback: str
    absa_result: ABSAResult
    submitted_at: str


class AnalyticsResponse(BaseModel):
    """Aggregated analytics across all or filtered responses."""

    total_responses: int
    total_surveys: int
    pedagogy_analytics: List[dict]


class PedagogyInfo(BaseModel):
    """Read-only schema returned by GET /api/pedagogies."""

    id: str
    name: str
    description: str


# ---------------------------------------------------------------------------
# Auth models (v3)
# ---------------------------------------------------------------------------


class GoogleAuthRequest(BaseModel):
    credential: str  # Google ID token


class AuthResponse(BaseModel):
    token: str
    name: str
    email: str
    already_submitted: bool = False


class TeacherLoginRequest(BaseModel):
    password: str


class TeacherAuthResponse(BaseModel):
    token: str
    role: str = "teacher"


class GeminiSummaryResponse(BaseModel):
    pedagogy_id: str
    pedagogy_name: str
    summary: str

