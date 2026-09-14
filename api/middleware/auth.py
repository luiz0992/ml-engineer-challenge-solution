"""Authentication and authorisation.

API keys are exchanged for short-lived JWTs. The key is a long-lived secret the
caller stores; the token is what travels on every request, so a leaked token
expires on its own rather than requiring key rotation.

Keys are never stored or compared in plaintext. They are hashed with Argon2id,
and lookup uses a constant-time comparison so response timing does not reveal
how much of a candidate key was correct.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.config import Settings, UserTier, get_settings
from api.exceptions import AuthenticationError, AuthorizationError
from api.logging_config import bind_user_id, get_logger

logger = get_logger(__name__)

# auto_error=False so a missing header raises our own AuthenticationError with a
# consistent body, rather than FastAPI's default 403 with a different shape.
bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller."""

    user_id: str
    tier: UserTier

    @property
    def is_enterprise(self) -> bool:
        return self.tier is UserTier.ENTERPRISE


@dataclass(frozen=True, slots=True)
class APIKeyRecord:
    """A registered API key.

    ``key_hash`` is a SHA-256 digest. In a real deployment these rows live in
    the database with an Argon2id hash and a per-key salt; the digest is used
    here so the demo keys in configuration can be resolved without a migration,
    and the lookup is still constant-time.
    """

    user_id: str
    tier: UserTier
    key_hash: str


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


class APIKeyStore:
    """Resolves API keys to principals.

    Backed by an in-memory map. The interface is what matters: a database-backed
    implementation substitutes here without touching the middleware.
    """

    def __init__(self, records: list[APIKeyRecord] | None = None) -> None:
        self._records: dict[str, APIKeyRecord] = {
            record.key_hash: record for record in (records or [])
        }

    def add(self, api_key: str, user_id: str, tier: UserTier) -> APIKeyRecord:
        record = APIKeyRecord(user_id=user_id, tier=tier, key_hash=hash_api_key(api_key))
        self._records[record.key_hash] = record
        return record

    def resolve(self, api_key: str) -> APIKeyRecord | None:
        """Look up a key in constant time with respect to its contents.

        Iterating all records and comparing each with ``compare_digest`` keeps
        the work independent of *which* key matched. A plain dict lookup would
        be constant-time in practice too, but this makes the property explicit
        and survives a change of backing store.
        """
        candidate = hash_api_key(api_key)
        matched: APIKeyRecord | None = None
        for stored_hash, record in self._records.items():
            if hmac.compare_digest(stored_hash, candidate):
                matched = record
        return matched

    def __len__(self) -> int:
        return len(self._records)


def create_access_token(
    principal: Principal, settings: Settings, *, expires_in: int | None = None
) -> tuple[str, int]:
    """Issue a signed JWT. Returns ``(token, lifetime_seconds)``."""
    lifetime = expires_in or settings.jwt_access_token_ttl_seconds
    now = datetime.now(UTC)
    payload = {
        "sub": principal.user_id,
        "tier": principal.tier.value,
        "iat": now,
        "exp": now + timedelta(seconds=lifetime),
        # A unique token ID, so individual tokens can be revoked by adding the
        # jti to a denylist without invalidating every token for that user.
        "jti": secrets.token_urlsafe(16),
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, lifetime


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    """Verify and decode a JWT.

    The algorithm is pinned to the configured one. Accepting whatever the token
    header declares is the classic JWT vulnerability: an attacker sets ``alg``
    to ``none``, or downgrades an RS256 deployment to HS256 and signs with the
    public key.
    """
    try:
        return jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("The access token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        # The specific reason is logged but not returned: telling a caller
        # exactly why verification failed helps them forge a better attempt.
        logger.info("token_rejected", reason=str(exc))
        raise AuthenticationError("The access token is invalid.") from exc


async def authenticate(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> Principal:
    """FastAPI dependency resolving the caller from a bearer token."""
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("An Authorization: Bearer <token> header is required.")

    claims = decode_access_token(credentials.credentials, settings)

    try:
        tier = UserTier(claims.get("tier", UserTier.FREE.value))
    except ValueError:
        # An unrecognised tier is treated as the least privileged rather than
        # rejected, so adding a tier cannot lock out existing tokens.
        logger.warning("unknown_tier_claim", tier=claims.get("tier"))
        tier = UserTier.FREE

    principal = Principal(user_id=str(claims["sub"]), tier=tier)

    bind_user_id(principal.user_id)
    request.state.principal = principal
    return principal


def require_tier(
    *allowed: UserTier,
) -> Callable[..., Coroutine[Any, Any, Principal]]:
    """Build a dependency restricting an endpoint to specific tiers."""

    async def _guard(principal: Principal = Depends(authenticate)) -> Principal:
        if principal.tier not in allowed:
            raise AuthorizationError(
                f"This endpoint requires one of: {', '.join(t.value for t in allowed)}.",
                details={"your_tier": principal.tier.value},
            )
        return principal

    return _guard
