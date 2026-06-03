"""Authentication helpers for Google OAuth and teacher login."""
import os
import logging
import hashlib
import secrets
from typing import Optional, Dict
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
TEACHER_PASSWORD = os.getenv("TEACHER_PASSWORD", "pedagogi_teacher_2024")
ALLOWED_DOMAIN = "rvce.edu.in"

# Simple in-memory session store (maps token -> email)
_student_sessions: Dict[str, str] = {}
_teacher_sessions: set = set()

def verify_google_token(token: str) -> Optional[Dict]:
    """Verify a Google token and return user info dict.

    Supports two flows:
    1. ID token (credential from GoogleLogin component)
    2. Access token (from useGoogleLogin popup flow) — verified via Google userinfo API

    Returns None if verification fails or email domain doesn't match.
    """
    # --- Try as ID token first ---
    try:
        idinfo = id_token.verify_oauth2_token(
            token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        email = idinfo.get("email", "")
        name = idinfo.get("name", "")
        if not email.endswith(f"@{ALLOWED_DOMAIN}"):
            logger.warning("Rejected login from non-college email: %s", email)
            return None
        return {"email": email, "name": name, "picture": idinfo.get("picture", "")}
    except Exception:
        pass  # Not an ID token, try as access token below

    # --- Try as access token (from popup OAuth flow) ---
    try:
        import urllib.request
        import json

        req = urllib.request.Request(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            userinfo = json.loads(resp.read())

        email = userinfo.get("email", "")
        name = userinfo.get("name", "")
        if not email.endswith(f"@{ALLOWED_DOMAIN}"):
            logger.warning("Rejected login from non-college email: %s", email)
            return None
        return {"email": email, "name": name, "picture": userinfo.get("picture", "")}
    except Exception as e:
        logger.error("Google token verification failed (both flows): %s", e)
        return None

def create_student_session(email: str) -> str:
    """Create a session token for a verified student."""
    token = secrets.token_hex(32)
    _student_sessions[token] = email
    return token

def get_student_email(session_token: str) -> Optional[str]:
    """Get student email from session token."""
    return _student_sessions.get(session_token)

def verify_teacher_password(password: str) -> bool:
    """Check if teacher password matches."""
    return password == TEACHER_PASSWORD

def create_teacher_session() -> str:
    """Create a session token for teacher."""
    token = secrets.token_hex(32)
    _teacher_sessions.add(token)
    return token

def is_teacher_session(token: str) -> bool:
    """Check if token is a valid teacher session."""
    return token in _teacher_sessions
