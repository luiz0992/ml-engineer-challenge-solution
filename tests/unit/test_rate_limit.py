"""Tests for the sliding-window rate limiter.

Exercised against ``fakeredis``, which implements real Redis semantics
including Lua evaluation, so the actual script runs rather than a stand-in.
Mocking Redis here would verify only that we call the methods we call, and
would say nothing about whether the window arithmetic is correct.
"""

from __future__ import annotations

import asyncio

import pytest

from api.config import Settings, UserTier
from api.exceptions import RateLimitExceededError
from api.middleware.rate_limit import RateLimiter

pytestmark = pytest.mark.unit


class TestQuotaEnforcement:
    async def test_admits_exactly_the_configured_quota(
        self, rate_limiter: RateLimiter, settings: Settings
    ) -> None:
        limit = settings.rate_limit_for(UserTier.FREE)

        for _ in range(limit):
            assert (await rate_limiter.check("alice", UserTier.FREE)).allowed

        # The next request is over quota.
        assert not (await rate_limiter.check("alice", UserTier.FREE)).allowed

    async def test_remaining_counts_down_to_zero(
        self, rate_limiter: RateLimiter, settings: Settings
    ) -> None:
        limit = settings.rate_limit_for(UserTier.FREE)
        observed = [
            (await rate_limiter.check("alice", UserTier.FREE)).remaining for _ in range(limit)
        ]
        assert observed == list(range(limit - 1, -1, -1))

    async def test_limits_differ_by_tier(self, rate_limiter: RateLimiter) -> None:
        free = await rate_limiter.check("alice", UserTier.FREE)
        pro = await rate_limiter.check("bob", UserTier.PRO)
        enterprise = await rate_limiter.check("carol", UserTier.ENTERPRISE)

        assert free.limit < pro.limit < enterprise.limit

    async def test_users_have_independent_windows(
        self, rate_limiter: RateLimiter, settings: Settings
    ) -> None:
        """One caller exhausting their quota must not affect another."""
        for _ in range(settings.rate_limit_for(UserTier.FREE) + 2):
            await rate_limiter.check("noisy", UserTier.FREE)

        assert (await rate_limiter.check("quiet", UserTier.FREE)).allowed

    async def test_same_user_on_different_tiers_is_tracked_separately(
        self, rate_limiter: RateLimiter, settings: Settings
    ) -> None:
        """Tier is part of the key, so an upgrade takes effect immediately."""
        for _ in range(settings.rate_limit_for(UserTier.FREE) + 1):
            await rate_limiter.check("alice", UserTier.FREE)

        assert (await rate_limiter.check("alice", UserTier.PRO)).allowed


class TestEnforceRaises:
    async def test_raises_with_actionable_detail(
        self, rate_limiter: RateLimiter, settings: Settings
    ) -> None:
        limit = settings.rate_limit_for(UserTier.FREE)
        for _ in range(limit):
            await rate_limiter.enforce("alice", UserTier.FREE)

        with pytest.raises(RateLimitExceededError) as exc_info:
            await rate_limiter.enforce("alice", UserTier.FREE)

        error = exc_info.value
        assert error.status_code == 429
        # A client cannot back off correctly without knowing how long to wait.
        assert error.retry_after_seconds > 0
        assert error.details["limit_per_minute"] == limit

    async def test_permits_traffic_under_quota(self, rate_limiter: RateLimiter) -> None:
        decision = await rate_limiter.enforce("alice", UserTier.PRO)
        assert decision.allowed


class TestHeaders:
    async def test_success_reports_limit_and_remaining(self, rate_limiter: RateLimiter) -> None:
        """Headers on success let clients self-throttle before being rejected."""
        headers = (await rate_limiter.check("alice", UserTier.PRO)).headers

        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "Retry-After" not in headers

    async def test_rejection_includes_retry_after(
        self, rate_limiter: RateLimiter, settings: Settings
    ) -> None:
        for _ in range(settings.rate_limit_for(UserTier.FREE) + 1):
            decision = await rate_limiter.check("alice", UserTier.FREE)

        assert not decision.allowed
        assert "Retry-After" in decision.headers
        assert decision.headers["X-RateLimit-Remaining"] == "0"


class TestAtomicity:
    async def test_concurrent_requests_cannot_exceed_the_limit(
        self, rate_limiter: RateLimiter, settings: Settings
    ) -> None:
        """The window check must be atomic.

        Issuing trim, count, and add as separate commands lets concurrent
        requests interleave between the count and the add, admitting more than
        the limit under exactly the load the limiter exists to control. The Lua
        script makes the sequence atomic; this fires four times the quota
        concurrently and asserts the excess is rejected.
        """
        limit = settings.rate_limit_for(UserTier.FREE)

        decisions = await asyncio.gather(
            *(rate_limiter.check("alice", UserTier.FREE) for _ in range(limit * 4))
        )
        admitted = sum(1 for decision in decisions if decision.allowed)

        assert admitted == limit, (
            f"{admitted} requests admitted against a limit of {limit}; "
            f"the window check is not atomic"
        )


class TestFailureModes:
    async def test_permits_traffic_when_redis_is_absent(self, settings: Settings) -> None:
        """With no Redis, the limiter fails open.

        Availability is preferred over strict enforcement for this service. A
        payment or authorisation path would make the opposite choice, which is
        why the behaviour is asserted rather than left implicit.
        """
        limiter = RateLimiter(None, settings)

        for _ in range(settings.rate_limit_for(UserTier.FREE) * 3):
            assert (await limiter.check("alice", UserTier.FREE)).allowed

    async def test_permits_traffic_when_redis_errors(self, settings: Settings, fake_redis) -> None:
        from redis.exceptions import ConnectionError as RedisConnectionError

        limiter = RateLimiter(fake_redis, settings)
        await limiter._ensure_script()

        async def _fail(*args: object, **kwargs: object) -> None:
            raise RedisConnectionError("connection lost")

        limiter._script = _fail  # type: ignore[assignment]

        decision = await limiter.check("alice", UserTier.FREE)
        assert decision.allowed, "a Redis outage must not reject traffic"

    async def test_reset_clears_the_window(
        self, rate_limiter: RateLimiter, settings: Settings
    ) -> None:
        limit = settings.rate_limit_for(UserTier.FREE)
        for _ in range(limit + 1):
            await rate_limiter.check("alice", UserTier.FREE)
        assert not (await rate_limiter.check("alice", UserTier.FREE)).allowed

        await rate_limiter.reset("alice", UserTier.FREE)
        assert (await rate_limiter.check("alice", UserTier.FREE)).allowed
