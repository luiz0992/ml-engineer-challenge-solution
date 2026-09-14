"""Redis-backed result cache.

Inference is deterministic for a given image, model version, and set of
options, so identical requests can be served from cache. The cache key is a
hash of the image bytes combined with everything that affects the result —
model name, version, backend, and request options. Omitting any of those would
let a cached result outlive the configuration that produced it, which is how a
model rollout silently keeps serving stale predictions.

**The cache is never allowed to fail a request.** Every operation is wrapped so
that a Redis outage degrades the service to uncached inference rather than
taking it down. A cache is an optimisation; treating it as a hard dependency
converts one failure domain into two.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from api.logging_config import get_logger

logger = get_logger(__name__)

#: Bumped when the cached payload shape changes, so a deploy cannot read old
#: entries with a new schema. Cheaper and safer than flushing.
CACHE_SCHEMA_VERSION = "v1"


class CacheService:
    """Async result cache with fail-open semantics."""

    def __init__(self, redis: Redis | None, *, ttl_seconds: int = 3600) -> None:
        self._redis = redis
        self.ttl_seconds = ttl_seconds
        self._available = redis is not None
        self.hits = 0
        self.misses = 0
        self.errors = 0

    @property
    def enabled(self) -> bool:
        return self._available and self._redis is not None

    @staticmethod
    def build_key(
        image_bytes: bytes,
        *,
        model_name: str,
        model_version: str,
        backend: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Derive a cache key from the image and everything affecting the result.

        SHA-256 of the image bytes, not of the decoded pixels: hashing bytes is
        far cheaper and two byte-identical uploads necessarily decode
        identically. Two visually identical images encoded differently will
        miss, which costs a recomputation but never returns a wrong answer.
        """
        digest = hashlib.sha256(image_bytes).hexdigest()
        option_repr = json.dumps(options or {}, sort_keys=True, separators=(",", ":"))
        option_digest = hashlib.sha256(option_repr.encode()).hexdigest()[:16]
        return (
            f"infer:{CACHE_SCHEMA_VERSION}:{model_name}:{model_version}"
            f":{backend}:{digest}:{option_digest}"
        )

    async def get(self, key: str) -> dict[str, Any] | None:
        """Return a cached payload, or ``None`` on miss, error, or disabled cache."""
        if not self.enabled:
            return None

        try:
            raw = await self._redis.get(key)  # type: ignore[union-attr]
        except RedisError as exc:
            self.errors += 1
            logger.warning("cache_get_failed", error=str(exc), key=key)
            return None

        if raw is None:
            self.misses += 1
            return None

        try:
            payload = json.loads(raw)
        except (ValueError, TypeError) as exc:
            # A corrupt entry must not poison the request; drop it and recompute.
            self.errors += 1
            logger.warning("cache_decode_failed", error=str(exc), key=key)
            await self.delete(key)
            return None

        self.hits += 1
        return payload

    async def set(self, key: str, value: dict[str, Any], *, ttl_seconds: int | None = None) -> bool:
        """Store a payload. Returns whether it was written."""
        if not self.enabled:
            return False

        try:
            await self._redis.set(  # type: ignore[union-attr]
                key,
                json.dumps(value, default=str),
                ex=ttl_seconds or self.ttl_seconds,
            )
        except (RedisError, TypeError, ValueError) as exc:
            # ValueError as well as TypeError: json.dumps raises ValueError on a
            # circular reference, and a cache that cannot serialise a value must
            # still not fail the request it was asked to speed up.
            self.errors += 1
            logger.warning("cache_set_failed", error=str(exc), key=key)
            return False
        return True

    async def delete(self, key: str) -> None:
        if not self.enabled:
            return
        try:
            await self._redis.delete(key)  # type: ignore[union-attr]
        except RedisError as exc:
            self.errors += 1
            logger.warning("cache_delete_failed", error=str(exc), key=key)

    async def ping(self) -> tuple[bool, str | None]:
        """Check connectivity, for the health endpoint."""
        if self._redis is None:
            return False, "cache not configured"
        try:
            await self._redis.ping()
        except RedisError as exc:
            return False, str(exc)
        return True, None

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, float | int | bool]:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": round(self.hit_rate, 4),
        }
