from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, Request
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt

from config import settings as cfg


def validate_auth_configuration():
    if cfg.BAKERY_ENV == "production" and cfg.JWT_SECRET_IS_EPHEMERAL:
        raise RuntimeError("JWT_SECRET must be configured in production")
    if (
        cfg.BAKERY_ENV == "production"
        and len(cfg.JWT_SECRET.encode("utf-8")) < 32
    ):
        raise RuntimeError("JWT_SECRET must contain at least 32 bytes in production")
    if cfg.BAKERY_ENV == "production" and cfg.ALLOW_LEGACY_PLAINTEXT_LOGIN:
        raise RuntimeError(
            "BAKERY_ALLOW_LEGACY_PLAINTEXT_LOGIN must be disabled in production"
        )


def create_access_token(username, role):
    validate_auth_configuration()
    return jwt.encode(
        {
            "sub": username,
            "role": role,
            "exp": datetime.utcnow() + timedelta(minutes=cfg.JWT_EXPIRE_MINUTES),
        },
        cfg.JWT_SECRET,
        algorithm=cfg.JWT_ALGORITHM,
    )


async def get_current_user(request: Request):
    validate_auth_configuration()
    scheme, token = get_authorization_scheme_param(
        request.headers.get("Authorization", "")
    )
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "Missing token")
    try:
        payload = jwt.decode(
            token,
            cfg.JWT_SECRET,
            algorithms=[cfg.JWT_ALGORITHM],
        )
    except JWTError as exc:
        raise HTTPException(401, "Invalid token") from exc
    if not payload.get("sub") or not payload.get("role"):
        raise HTTPException(401, "Invalid token")
    return payload


async def require_manager(user=Depends(get_current_user)):
    if user.get("role") != "manager":
        raise HTTPException(403, "Manager only")
    return user
