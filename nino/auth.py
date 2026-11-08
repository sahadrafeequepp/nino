"""Multi-user authentication, session cookies, password hashing, and lockout management for Nino."""

from datetime import datetime, timedelta, timezone
import hashlib
import re
import secrets
from typing import Any
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from nino.config import Settings, get_settings
from nino.memory.db import MemoryDB
from nino.security.lockout import get_lockout_manager

security_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Derive secure PBKDF2-HMAC-SHA256 password hash and salt."""
    if not salt:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return key.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Constant-time password hash verification."""
    computed, _ = hash_password(password, salt=salt)
    return secrets.compare_digest(computed, password_hash)


def create_user(memory_db: MemoryDB, username: str, password: str) -> dict[str, Any]:
    """Create a new user account with unique username and hashed password."""
    clean_username = username.strip()
    if not (3 <= len(clean_username) <= 64):
        raise ValueError("Username must be between 3 and 64 characters.")
    if not re.match(r"^[a-zA-Z0-9_\-\.\@ ]+$", clean_username):
        raise ValueError("Username may only contain letters, numbers, spaces, dots, hyphens, @, and underscores.")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters long.")

    with memory_db.get_connection() as conn:
        cur = conn.execute("SELECT id FROM users WHERE username = ?", (clean_username,))
        if cur.fetchone():
            raise ValueError(f"Username '{clean_username}' is already taken.")

        user_id = "u_" + secrets.token_hex(8)
        p_hash, salt = hash_password(password)
        conn.execute(
            "INSERT INTO users (id, username, password_hash, salt) VALUES (?, ?, ?, ?)",
            (user_id, clean_username, p_hash, salt),
        )
    return {"id": user_id, "username": clean_username}


def authenticate_user(memory_db: MemoryDB, username: str, password: str) -> dict[str, Any] | None:
    """Verify username and password against users table."""
    clean_username = username.strip()
    with memory_db.get_connection() as conn:
        cur = conn.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
            (clean_username,),
        )
        row = cur.fetchone()
        if not row:
            return None
        if not verify_password(password, row["password_hash"], row["salt"]):
            return None
        return {"id": row["id"], "username": row["username"], "is_admin": (row["username"].lower() == "admin")}


def create_user_session(memory_db: MemoryDB, user_id: str, username: str, duration_days: int = 30) -> str:
    """Create an authenticated user session token persisted in user_sessions table."""
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=duration_days)).isoformat()
    with memory_db.get_connection() as conn:
        if user_id == "admin":
            conn.execute(
                "INSERT OR IGNORE INTO users (id, username, password_hash, salt) VALUES ('admin', 'admin', 'system_admin', 'system_admin')"
            )
        conn.execute(
            "INSERT INTO user_sessions (token, user_id, username, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, username, expires_at),
        )
    return token


def get_user_by_session_token(memory_db: MemoryDB, token: str) -> dict[str, Any] | None:
    """Look up active user session by session token."""
    if not token or not token.strip():
        return None
    with memory_db.get_connection() as conn:
        cur = conn.execute(
            "SELECT token, user_id, username, expires_at FROM user_sessions WHERE token = ?",
            (token.strip(),),
        )
        row = cur.fetchone()
        if not row:
            return None
        expires_at_str = row["expires_at"]
        try:
            exp_dt = datetime.fromisoformat(expires_at_str)
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > exp_dt:
                conn.execute("DELETE FROM user_sessions WHERE token = ?", (token.strip(),))
                return None
        except Exception:
            pass
        return {"id": row["user_id"], "username": row["username"], "is_admin": (row["username"] == "admin")}


def delete_user_session(memory_db: MemoryDB, token: str) -> bool:
    """Revoke and delete a user session."""
    if not token:
        return False
    with memory_db.get_connection() as conn:
        cur = conn.execute("DELETE FROM user_sessions WHERE token = ?", (token.strip(),))
        return cur.rowcount > 0


def get_memory_db_from_request(request: Request, settings: Settings) -> MemoryDB:
    db = None
    if request:
        try:
            db = getattr(request.app.state, "memory_db", None)
        except Exception:
            db = None
    if not db:
        db = MemoryDB(db_path=settings.db_path)
    return db


def verify_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
    settings: Settings = Depends(get_settings),
) -> str:
    """Verify authentication via Bearer token header or 'nino_session' HTTP cookie.

    Raises:
        HTTPException: 429 Too Many Requests if client IP is locked out.
        HTTPException: 401 Unauthorized if token/cookie is missing or invalid.
    """
    client_ip = request.client.host if (request and request.client) else "127.0.0.1"
    lockout_mgr = None
    if request:
        try:
            lockout_mgr = getattr(request.app.state, "lockout_manager", None)
        except Exception:
            lockout_mgr = None
    if not lockout_mgr:
        lockout_mgr = get_lockout_manager()

    # 1. Check if IP is locked out
    is_locked, remaining = lockout_mgr.is_locked_out(client_ip)
    if is_locked:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed authentication attempts. Locked out for {remaining} seconds.",
        )

    db = get_memory_db_from_request(request, settings)

    # 2. Check Bearer credentials if provided
    provided_token = credentials.credentials if (credentials and credentials.credentials) else None
    if provided_token:
        # A. Constant-time comparison with system admin token
        expected_token = settings.auth_token.encode("utf-8")
        if secrets.compare_digest(provided_token.encode("utf-8"), expected_token):
            lockout_mgr.record_success(client_ip)
            request.state.user = {"id": "admin", "username": "admin", "is_admin": True}
            return provided_token

        # B. Check user_sessions database
        user = get_user_by_session_token(db, provided_token)
        if user:
            lockout_mgr.record_success(client_ip)
            request.state.user = user
            return provided_token

    # 3. Check session cookie
    cookie_token = request.cookies.get("nino_session")
    if cookie_token:
        user = get_user_by_session_token(db, cookie_token)
        if user:
            lockout_mgr.record_success(client_ip)
            request.state.user = user
            return cookie_token

    # 4. Neither valid -> Record failure and raise 401
    lockout_mgr.record_failure(client_ip)
    if provided_token or cookie_token:
        err_msg = "Invalid authentication token."
    else:
        err_msg = "Authorization header or session cookie missing or empty."
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=err_msg,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    request: Request,
    token: str = Depends(verify_token),
) -> dict[str, Any]:
    """Retrieve the currently authenticated user from request state."""
    user = getattr(request.state, "user", None)
    if not user:
        user = {"id": "admin", "username": "admin", "is_admin": True}
    return user


def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any] | None:
    """Retrieve authenticated user if valid token/cookie present, without raising 401."""
    try:
        token = credentials.credentials if (credentials and credentials.credentials) else None
        db = get_memory_db_from_request(request, settings)
        if token:
            if secrets.compare_digest(token.encode("utf-8"), settings.auth_token.encode("utf-8")):
                return {"id": "admin", "username": "admin", "is_admin": True}
            user = get_user_by_session_token(db, token)
            if user:
                return user
        cookie_token = request.cookies.get("nino_session")
        if cookie_token:
            return get_user_by_session_token(db, cookie_token)
    except Exception:
        pass
    return None
