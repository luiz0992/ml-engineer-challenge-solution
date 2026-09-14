"""Per-tier rate limiting.

Implements a sliding-window counter in Redis. The window is a sorted set of
request timestamps per caller: expired entries are trimmed, the remainder
counted, and the request admitted or rejected.

A sliding window is used rather than a fixed window because fixed windows admit
double the intended rate at a boundary — a caller can spend a full quota in the
last instant of one window and another immediately after. The sorted-set
approach costs one extra Redis operation and removes that burst entirely.

The whole sequence runs as a single Lua script so it is atomic. Issuing trim,
count, and add as separate commands lets concurrent requests interleave between
the count and the add, and the limit is then exceeded under exactly the load it
exists to control.

**Fails open.** If Redis is unreachable the request is allowed. For this
service, availability outweighs strict quota enforcement; a payment or
authorisation path would make the opposite choice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.commands.core import AsyncScript
from redis.exceptions import RedisError

from api.config import Settings, UserTier
from api.exceptions import RateLimitExceededError
from api.logging_config import get_logger

logger = get_logger(__name__)

WINDOW_SECONDS = 60

#: Atomic sliding-window check.
#:   KEYS[1] - the caller's window key
#:   ARGV[1] - current time (ms)
#:   ARGV[2] - window length (ms)
#:   ARGV[3] - limit
#:   ARGV[4] - unique member for this request
#: Returns {allowed, current_count, oldest_timestamp_ms}
_SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('ZADD', key, now, member)
    -- Expire slightly after the window so an idle key cleans itself up.
    redis.call('PEXPIRE', key, window + 1000)
    return {1, count + 1, 0}
end

local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
local oldest_ts = 0
if oldest[2] then oldest_ts = tonumber(oldest[2]) end
return {0, count, oldest_ts}
"""


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Outcome of a rate-limit check, used to populate response headers."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int

    @property
    def headers(self) -> dict[str, str]:
        """Standard rate-limit headers.

        Returned on success as well as rejection so clients can self-throttle
        before being rejected, rather than discovering the limit by hitting it.
        """
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
        }
        if not self.allowed:
            headers["Retry-After"] = str(self.retry_after_seconds)
        return headers


class RateLimiter:
    """Sliding-window rate limiter backed by Redis."""

    def __init__(self, redis: Redis | None, settings: Settings) -> None:
        self._redis = redis
        self.settings = settings
        self._script: AsyncScript | None = None

    async def _ensure_script(self) -> None:
        if self._script is None and self._redis is not None:
            self._script = self._redis.register_script(_SLIDING_WINDOW_SCRIPT)

    async def check(self, user_id: str, tier: UserTier) -> RateLimitDecision:
        """Record a request and decide whether it is permitted."""
        limit = self.settings.rate_limit_for(tier)

        if self._redis is None:
            # No Redis configured: enforcement is impossible, so permit and
            # report the quota honestly rather than pretending to enforce it.
            return RateLimitDecision(True, limit, limit, 0)

        await self._ensure_script()
        now_ms = int(time.time() * 1000)
        window_ms = WINDOW_SECONDS * 1000
        key = f"ratelimit:{tier.value}:{user_id}"
        member = f"{now_ms}:{time.monotonic_ns()}"

        try:
            allowed, count, oldest_ts = await self._script(  # type: ignore[misc]
                keys=[key], args=[now_ms, window_ms, limit, member]
            )
        except RedisError as exc:
            logger.warning("rate_limit_unavailable", error=str(exc), user_id=user_id)
            return RateLimitDecision(True, limit, limit, 0)

        if int(allowed) == 1:
            return RateLimitDecision(True, limit, limit - int(count), 0)

        # Retry when the oldest request in the window falls out of it.
        retry_after = max(1, int((int(oldest_ts) + window_ms - now_ms) / 1000) + 1)
        return RateLimitDecision(False, limit, 0, retry_after)

    async def enforce(self, user_id: str, tier: UserTier) -> RateLimitDecision:
        """Check the limit, raising if the caller is over quota."""
        decision = await self.check(user_id, tier)
        if not decision.allowed:
            logger.info(
                "rate_limit_exceeded",
                user_id=user_id,
                tier=tier.value,
                limit=decision.limit,
            )
            raise RateLimitExceededError(
                f"Rate limit of {decision.limit} requests per minute exceeded for "
                f"the {tier.value} tier.",
                retry_after_seconds=decision.retry_after_seconds,
                limit=decision.limit,
            )
        return decision

    async def reset(self, user_id: str, tier: UserTier) -> None:
        """Clear a caller's window. Used by tests and support tooling."""
        if self._redis is None:
            return
        try:
            await self._redis.delete(f"ratelimit:{tier.value}:{user_id}")
        except RedisError as exc:
            logger.warning("rate_limit_reset_failed", error=str(exc))
