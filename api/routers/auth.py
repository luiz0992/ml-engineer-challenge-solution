"""Token issuance."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.config import Settings, get_settings
from api.exceptions import AuthenticationError
from api.logging_config import get_logger
from api.middleware.auth import Principal, create_access_token
from api.models.responses import ErrorResponse, TokenResponse
from api.models.schemas import TokenRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Exchange an API key for an access token",
    responses={401: {"model": ErrorResponse, "description": "Unknown API key"}},
)
async def issue_token(
    payload: TokenRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Exchange a long-lived API key for a short-lived bearer token.

    The token carries the caller's tier, which determines their rate limit.
    Short lifetimes mean a leaked token stops working on its own, without
    needing the underlying key to be rotated.
    """
    store = request.app.state.api_key_store
    record = store.resolve(payload.api_key)

    if record is None:
        # Logged without the key itself. The failure reason returned to the
        # caller is deliberately identical for unknown and malformed keys, so
        # responses cannot be used to probe which keys exist.
        logger.info(
            "token_request_rejected", client=request.client.host if request.client else None
        )
        raise AuthenticationError("The API key is not recognised.")

    principal = Principal(user_id=record.user_id, tier=record.tier)
    token, lifetime = create_access_token(principal, settings)

    logger.info("token_issued", user_id=record.user_id, tier=record.tier.value)

    return TokenResponse(
        access_token=token,
        expires_in=lifetime,
        tier=record.tier.value,
    )
