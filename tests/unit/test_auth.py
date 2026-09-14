"""Tests for authentication and token handling."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from api.config import Settings, UserTier
from api.exceptions import AuthenticationError, AuthorizationError
from api.middleware.auth import (
    APIKeyStore,
    Principal,
    create_access_token,
    decode_access_token,
    hash_api_key,
)

pytestmark = pytest.mark.unit


class TestAPIKeyStore:
    def test_resolves_a_registered_key(self) -> None:
        store = APIKeyStore()
        store.add("secret-key-abc", user_id="alice", tier=UserTier.PRO)

        record = store.resolve("secret-key-abc")
        assert record is not None
        assert record.user_id == "alice"
        assert record.tier is UserTier.PRO

    def test_returns_none_for_unknown_key(self) -> None:
        store = APIKeyStore()
        store.add("secret-key-abc", user_id="alice", tier=UserTier.PRO)
        assert store.resolve("some-other-key") is None

    def test_never_stores_the_plaintext_key(self) -> None:
        """A store dump must not reveal usable credentials.

        If the process memory or a serialised store is ever exposed, hashed
        keys cannot be replayed against the API.
        """
        store = APIKeyStore()
        store.add("super-secret-value", user_id="alice", tier=UserTier.FREE)

        serialised = repr(store.__dict__)
        assert "super-secret-value" not in serialised
        assert hash_api_key("super-secret-value") in serialised

    def test_rejects_a_prefix_of_a_valid_key(self) -> None:
        """Partial matches must fail, guarding against prefix-probing."""
        store = APIKeyStore()
        store.add("abcdefghijklmnop", user_id="alice", tier=UserTier.FREE)

        assert store.resolve("abcdefgh") is None
        assert store.resolve("abcdefghijklmnopq") is None

    def test_lookup_time_does_not_depend_on_key_similarity(self) -> None:
        """Timing must not reveal how much of a candidate key was correct.

        A comparison that short-circuits on the first differing byte lets an
        attacker recover a key one byte at a time. The margin here is
        deliberately loose: this asserts the absence of an order-of-magnitude
        signal, not precise timing, so it does not flake on a noisy machine.
        """
        store = APIKeyStore()
        store.add("a" * 64, user_id="alice", tier=UserTier.FREE)

        def measure(candidate: str, iterations: int = 300) -> float:
            start = time.perf_counter()
            for _ in range(iterations):
                store.resolve(candidate)
            return time.perf_counter() - start

        nearly_correct = measure("a" * 63 + "b")
        entirely_wrong = measure("z" * 64)

        ratio = max(nearly_correct, entirely_wrong) / max(min(nearly_correct, entirely_wrong), 1e-9)
        assert ratio < 3.0, f"lookup timing varies with key similarity (ratio {ratio:.1f})"


class TestTokenLifecycle:
    def test_round_trips_identity_and_tier(self, settings: Settings) -> None:
        principal = Principal(user_id="alice", tier=UserTier.ENTERPRISE)
        token, lifetime = create_access_token(principal, settings)

        claims = decode_access_token(token, settings)
        assert claims["sub"] == "alice"
        assert claims["tier"] == "enterprise"
        assert lifetime == settings.jwt_access_token_ttl_seconds

    def test_tokens_are_unique_per_issue(self, settings: Settings) -> None:
        """Each token carries a distinct jti, so one can be revoked alone."""
        principal = Principal(user_id="alice", tier=UserTier.FREE)
        first, _ = create_access_token(principal, settings)
        second, _ = create_access_token(principal, settings)

        assert first != second
        assert (
            decode_access_token(first, settings)["jti"]
            != decode_access_token(second, settings)["jti"]
        )

    def test_rejects_expired_token(self, settings: Settings) -> None:
        expired = jwt.encode(
            {
                "sub": "alice",
                "tier": "pro",
                "exp": datetime.now(UTC) - timedelta(seconds=1),
            },
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
        with pytest.raises(AuthenticationError, match="expired"):
            decode_access_token(expired, settings)

    def test_rejects_token_signed_with_another_key(self, settings: Settings) -> None:
        forged = jwt.encode(
            {
                "sub": "attacker",
                "tier": "enterprise",
                "exp": datetime.now(UTC) + timedelta(hours=1),
            },
            "a-completely-different-secret-of-sufficient-length-for-hs256",
            algorithm="HS256",
        )
        with pytest.raises(AuthenticationError):
            decode_access_token(forged, settings)

    def test_rejects_unsigned_token(self, settings: Settings) -> None:
        """The alg=none attack.

        An attacker strips the signature and sets the algorithm to ``none``.
        A verifier that trusts the token's own header accepts it. Pinning the
        algorithm is what prevents this.
        """
        unsigned = jwt.encode(
            {
                "sub": "attacker",
                "tier": "enterprise",
                "exp": datetime.now(UTC) + timedelta(hours=1),
            },
            key="",
            algorithm="none",
        )
        with pytest.raises(AuthenticationError):
            decode_access_token(unsigned, settings)

    def test_rejects_token_without_expiry(self, settings: Settings) -> None:
        """A token with no exp would be valid forever if accepted."""
        eternal = jwt.encode(
            {"sub": "alice", "tier": "pro"},
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
        with pytest.raises(AuthenticationError):
            decode_access_token(eternal, settings)

    def test_error_message_does_not_explain_the_failure(self, settings: Settings) -> None:
        """Verification failures must not help an attacker refine an attempt."""
        with pytest.raises(AuthenticationError) as exc_info:
            decode_access_token("not-even-a-jwt", settings)

        message = exc_info.value.message.lower()
        assert "signature" not in message
        assert "algorithm" not in message


class TestTierEnforcement:
    async def test_allows_permitted_tier(self, principal_factory) -> None:
        from api.middleware.auth import require_tier

        guard = require_tier(UserTier.PRO, UserTier.ENTERPRISE)
        principal = principal_factory(tier=UserTier.PRO)
        assert await guard(principal=principal) is principal

    async def test_rejects_insufficient_tier(self, principal_factory) -> None:
        from api.middleware.auth import require_tier

        guard = require_tier(UserTier.ENTERPRISE)
        with pytest.raises(AuthorizationError) as exc_info:
            await guard(principal=principal_factory(tier=UserTier.FREE))

        assert exc_info.value.status_code == 403
        assert exc_info.value.details["your_tier"] == "free"
