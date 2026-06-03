"""
FastAPI application for the Pedagogical Intelligence System.

Endpoints
---------
* ``GET  /``                       – welcome message
* ``GET  /api/pedagogies``         – list supported pedagogy types
* ``POST /api/survey/submit``      – submit anonymous survey + run ABSA
* ``GET  /api/responses``          – list all stored responses
* ``GET  /api/responses/{pid}``    – responses for one pedagogy
* ``GET  /api/analytics``          – aggregated analytics (all)
* ``GET  /api/analytics/{pid}``    – analytics for one pedagogy
* ``GET  /api/health``             – health check
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from auth import (
    verify_google_token,
    create_student_session,
    get_student_email,
    verify_teacher_password,
    create_teacher_session,
    is_teacher_session,
)
from database import (
    check_email_submitted,
    get_all_responses,
    get_analytics,
    get_response_count,
    get_responses_by_pedagogy,
    init_db,
    insert_survey,
)
from gemini_service import invalidate_cache
from models import (
    ABSAResult,
    AnalyticsResponse,
    GoogleAuthRequest,
    PEDAGOGIES,
    PedagogyInfo,
    ResponseRecord,
    SurveySubmission,
    TeacherLoginRequest,
    pedagogy_name,
)
from nlp_pipeline import run_absa_pipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database on startup."""
    logger.info("Initialising database …")
    init_db()
    logger.info("Database ready.")
    yield  # application runs here
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pedagogical Intelligence System API",
    description=(
        "Enterprise-grade ABSA-powered survey analysis for pedagogy "
        "effectiveness evaluation – RV College of Engineering, Bengaluru."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# CORS – allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> Dict[str, str]:
    """Welcome message."""
    return {
        "message": "Welcome to the Pedagogical Intelligence System API v3.0",
        "docs": "/docs",
    }


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Quick health probe."""
    return {
        "status": "healthy",
        "total_responses": get_response_count(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/pedagogies", response_model=List[PedagogyInfo])
def list_pedagogies() -> List[Dict[str, str]]:
    """Return the canonical list of pedagogy types."""
    return PEDAGOGIES  # type: ignore[return-value]


# ---- Survey submission ---------------------------------------------------


@app.post("/api/survey/submit")
def submit_survey(
    submission: SurveySubmission,
    authorization: str = Header(None),
) -> Dict[str, Any]:
    """Accept a survey, run ABSA on every feedback, persist results.

    Returns the survey ID and the per-pedagogy ABSA results so the frontend
    can show instant feedback to the student.
    """
    if not submission.responses:
        raise HTTPException(status_code=400, detail="No responses provided.")

    # Extract student email from session token (if authenticated)
    email = None
    if authorization:
        token = authorization.replace("Bearer ", "")
        email = get_student_email(token)
        if email and check_email_submitted(email):
            raise HTTPException(
                status_code=403,
                detail="You have already submitted a survey.",
            )

    survey_id = uuid.uuid4().hex[:12]
    submitted_at = (
        submission.submitted_at or datetime.now(timezone.utc)
    ).isoformat()

    results: List[Dict[str, Any]] = []

    for rating in submission.responses:
        # Run the NLP pipeline
        absa = run_absa_pipeline(rating.feedback)

        record = {
            "pedagogy_id": rating.pedagogy_id,
            "pedagogy_name": pedagogy_name(rating.pedagogy_id),
            "effectiveness": rating.effectiveness,
            "engagement": rating.engagement,
            "clarity": rating.clarity,
            "relevance": rating.relevance,
            "feedback": rating.feedback,
            "absa_result": absa,
        }
        results.append(record)

    # Persist in a single transaction
    try:
        insert_survey(survey_id, submitted_at, results, student_email=email)
    except Exception as exc:
        logger.exception("Failed to persist survey %s", survey_id)
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {exc}",
        ) from exc

    # Invalidate Gemini summary cache so new data is reflected
    invalidate_cache()

    return {
        "survey_id": survey_id,
        "submitted_at": submitted_at,
        "results": results,
    }


# ---- Response listing ----------------------------------------------------


@app.get("/api/responses", response_model=List[ResponseRecord])
def list_responses() -> List[Dict[str, Any]]:
    """Return all stored responses (newest first)."""
    return get_all_responses()


@app.get("/api/responses/{pedagogy_id}", response_model=List[ResponseRecord])
def list_responses_by_pedagogy(pedagogy_id: str) -> List[Dict[str, Any]]:
    """Return responses for a single pedagogy type."""
    rows = get_responses_by_pedagogy(pedagogy_id)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No responses found for pedagogy '{pedagogy_id}'.",
        )
    return rows


# ---- Analytics -----------------------------------------------------------


@app.get("/api/analytics", response_model=AnalyticsResponse)
def analytics_all() -> Dict[str, Any]:
    """Aggregated analytics across all pedagogies."""
    return get_analytics()


@app.get("/api/analytics/{pedagogy_id}", response_model=AnalyticsResponse)
def analytics_by_pedagogy(pedagogy_id: str) -> Dict[str, Any]:
    """Detailed analytics for one pedagogy type."""
    data = get_analytics(pedagogy_id=pedagogy_id)
    if data["total_responses"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data for pedagogy '{pedagogy_id}'.",
        )
    return data


# ---- Auth endpoints (v3) -------------------------------------------------


@app.post("/api/auth/google")
def google_login(req: GoogleAuthRequest) -> Dict[str, Any]:
    """Student Google login – only @rvce.edu.in emails allowed."""
    user_info = verify_google_token(req.credential)
    if not user_info:
        raise HTTPException(
            status_code=403,
            detail="Invalid token or non-college email. Only @rvce.edu.in emails allowed.",
        )
    email = user_info["email"]
    already_submitted = check_email_submitted(email)
    session_token = create_student_session(email)
    return {
        "token": session_token,
        "name": user_info["name"],
        "email": email,
        "already_submitted": already_submitted,
    }


@app.post("/api/auth/email")
def email_login(req: Dict[str, Any]) -> Dict[str, Any]:
    """Simple college email login — validates @rvce.edu.in domain."""
    email = (req.get("email") or "").strip().lower()
    name = (req.get("name") or "").strip()

    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")
    if not email.endswith("@rvce.edu.in"):
        raise HTTPException(
            status_code=403,
            detail="Only @rvce.edu.in college emails are allowed.",
        )
    if not name:
        # Extract name from email prefix
        name = email.split("@")[0].replace(".", " ").title()

    already_submitted = check_email_submitted(email)
    session_token = create_student_session(email)
    return {
        "token": session_token,
        "name": name,
        "email": email,
        "already_submitted": already_submitted,
    }


@app.post("/api/teacher/login")
def teacher_login(req: TeacherLoginRequest) -> Dict[str, Any]:
    """Teacher password-based login."""
    if not verify_teacher_password(req.password):
        raise HTTPException(status_code=403, detail="Invalid password.")
    token = create_teacher_session()
    return {"token": token, "role": "teacher"}


@app.get("/api/teacher/verify")
def verify_teacher(authorization: str = Header(None)) -> Dict[str, Any]:
    """Verify a teacher session token."""
    if not authorization or not is_teacher_session(
        authorization.replace("Bearer ", "")
    ):
        raise HTTPException(status_code=403, detail="Invalid teacher session.")
    return {"valid": True}


# ---- Gemini AI summary (v3) ----------------------------------------------


@app.get("/api/summary/{pedagogy_id}")
def get_summary(pedagogy_id: str) -> Dict[str, Any]:
    """Generate a Gemini AI summary for a pedagogy."""
    from gemini_service import generate_pedagogy_summary

    analytics = get_analytics(pedagogy_id=pedagogy_id)
    if analytics["total_responses"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data for pedagogy '{pedagogy_id}'.",
        )

    ped_stats = analytics["pedagogy_analytics"][0]
    responses = get_responses_by_pedagogy(pedagogy_id)
    feedback_list = [r["feedback"] for r in responses]

    ped_info = next((p for p in PEDAGOGIES if p["id"] == pedagogy_id), {})

    summary = generate_pedagogy_summary(
        pedagogy_name=ped_stats["pedagogy_name"],
        pedagogy_description=ped_info.get("description", ""),
        feedback_list=feedback_list,
        avg_ratings=ped_stats,
        sentiment_distribution=ped_stats["sentiment_distribution"],
        top_aspects=ped_stats["top_aspects"],
        response_count=ped_stats["count"],
    )
    return {
        "pedagogy_id": pedagogy_id,
        "pedagogy_name": ped_stats["pedagogy_name"],
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
