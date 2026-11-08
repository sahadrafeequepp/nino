"""Authentication and User Account API Endpoints for Nino."""

import secrets
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from nino.auth import (
    authenticate_user,
    create_user,
    create_user_session,
    delete_user_session,
    get_memory_db_from_request,
    get_optional_user,
)
from nino.config import Settings, get_settings
from nino.security.lockout import get_lockout_manager

router = APIRouter(prefix="/api/auth", tags=["auth"])


class AuthRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="User name (letters, numbers, spaces, -, _, ., @)")
    password: str = Field(..., min_length=4, max_length=128, description="User password")


class AuthLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64, description="Username or 'admin'")
    password: str = Field(..., min_length=1, max_length=128, description="Password or master auth token")


COOKIE_NAME = "nino_session"
COOKIE_MAX_AGE = 30 * 86400  # 30 days in seconds


def set_auth_cookie(response: Response, token: str) -> None:
    """Attach the HttpOnly authenticated session cookie."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )


@router.post("/register", summary="Register a new personal user account")
async def register(
    payload: AuthRegisterRequest,
    response: Response,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Register a new user account, generate an active session, and set session cookie."""
    db = get_memory_db_from_request(request, settings)
    try:
        user = create_user(db, payload.username, payload.password)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    token = create_user_session(db, user_id=user["id"], username=user["username"])
    set_auth_cookie(response, token)
    return {
        "status": "ok",
        "message": f"Account '{user['username']}' registered successfully.",
        "user": user,
        "token": token,
    }


@router.post("/login", summary="Sign in with username and password")
async def login(
    payload: AuthLoginRequest,
    response: Response,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Authenticate with username & password, generate a session, and set cookie."""
    client_ip = request.client.host if (request and request.client) else "127.0.0.1"
    lockout_mgr = None
    if request:
        try:
            lockout_mgr = getattr(request.app.state, "lockout_manager", None)
        except Exception:
            lockout_mgr = None
    if not lockout_mgr:
        lockout_mgr = get_lockout_manager()

    is_locked, remaining = lockout_mgr.is_locked_out(client_ip)
    if is_locked:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Locked out for {remaining} seconds.",
        )

    db = get_memory_db_from_request(request, settings)
    user: dict[str, Any] | None = None

    # Check for master admin login
    if payload.username.lower() == "admin" and secrets.compare_digest(
        payload.password.encode("utf-8"), settings.auth_token.encode("utf-8")
    ):
        user = {"id": "admin", "username": "admin", "is_admin": True}
    else:
        user = authenticate_user(db, payload.username, payload.password)

    if not user:
        lockout_mgr.record_failure(client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    lockout_mgr.record_success(client_ip)
    token = create_user_session(db, user_id=user["id"], username=user["username"])
    set_auth_cookie(response, token)
    return {
        "status": "ok",
        "message": f"Welcome back, {user['username']}!",
        "user": user,
        "token": token,
    }


@router.post("/logout", summary="Sign out and clear session cookie")
async def logout(
    response: Response,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Revoke session token in DB and remove browser cookie."""
    token = request.cookies.get(COOKIE_NAME)
    if token:
        db = get_memory_db_from_request(request, settings)
        delete_user_session(db, token)
    response.delete_cookie(key=COOKIE_NAME, path="/")
    return {"status": "ok", "message": "Signed out successfully."}


@router.get("/me", summary="Get details of currently authenticated user")
async def get_me(
    user: dict[str, Any] | None = Depends(get_optional_user),
):
    """Return user info if authenticated via cookie or token, else authenticated=false."""
    if user:
        return {
            "authenticated": True,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "is_admin": user.get("is_admin", False),
            },
        }
    return {
        "authenticated": False,
        "user": None,
    }

