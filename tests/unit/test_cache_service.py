"""Tests for the result cache.

The behaviour that matters most is what happens when Redis misbehaves. A cache
is an optimisation, and the service must never fail a request because of one —
so every failure path is asserted explicitly rather than assumed.
"""

from __future__ import annotations

from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from api.services.cache_service import CACHE_SCHEMA_VERSION, CacheService

pytestmark = pytest.mark.unit


class TestCacheKeys:
    def test_identical_inputs_produce_identical_keys(self) -> None:
        arguments: dict[str, Any] = {
            "model_name": "classifier",
            "model_version": "v1",
            "backend": "onnx",
            "options": {"top_k": 5},
        }
        assert CacheService.build_key(b"image", **arguments) == CacheService.build_key(
            b"image", **arguments
        )

    def test_different_images_produce_different_keys(self) -> None:
        arguments: dict[str, Any] = {
            "model_name": "c",
            "model_version": "v1",
            "backend": "onnx",
        }
        assert CacheService.build_key(b"image-a", **arguments) != CacheService.build_key(
            b"image-b", **arguments
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model_version", "v2"),
            ("model_name", "other-model"),
            ("backend", "tensorrt"),
        ],
    )
    def test_model_identity_is_part_of_the_key(self, field: str, value: str) -> None:
        """A model change must invalidate cached results.

        Without this, a rollout keeps serving predictions produced by the
        previous version until the TTL expires — a silent correctness failure
        that looks exactly like a successful deploy.
        """
        base: dict[str, Any] = {
            "model_name": "classifier",
            "model_version": "v1",
            "backend": "onnx",
        }
        assert CacheService.build_key(b"image", **base) != CacheService.build_key(
            b"image", **{**base, field: value}
        )

    def test_request_options_are_part_of_the_key(self) -> None:
        """Different options must not collide.

        A top_k=1 result served to a top_k=10 request would return four fewer
        predictions than asked for.
        """
        base: dict[str, Any] = {
            "model_name": "c",
            "model_version": "v1",
            "backend": "onnx",
        }
        assert CacheService.build_key(
            b"image", **base, options={"top_k": 1}
        ) != CacheService.build_key(b"image", **base, options={"top_k": 10})

    def test_option_ordering_does_not_affect_the_key(self) -> None:
        """Semantically identical options must hit the same entry."""
        base: dict[str, Any] = {
            "model_name": "c",
            "model_version": "v1",
            "backend": "onnx",
        }
        assert CacheService.build_key(
            b"image", **base, options={"top_k": 5, "probs": True}
        ) == CacheService.build_key(b"image", **base, options={"probs": True, "top_k": 5})

    def test_key_carries_a_schema_version(self) -> None:
        """A payload shape change must not be read with the new schema."""
        key = CacheService.build_key(b"image", model_name="c", model_version="v1", backend="onnx")
        assert f":{CACHE_SCHEMA_VERSION}:" in key

    def test_key_does_not_embed_raw_image_bytes(self) -> None:
        key = CacheService.build_key(
            b"\x89PNG-secret-content", model_name="c", model_version="v1", backend="onnx"
        )
        assert "secret-content" not in key


class TestCacheOperations:
    async def test_stores_and_retrieves(self, cache_service: CacheService) -> None:
        await cache_service.set("k", {"predictions": [{"label": "cat"}]})
        assert await cache_service.get("k") == {"predictions": [{"label": "cat"}]}

    async def test_missing_key_returns_none(self, cache_service: CacheService) -> None:
        assert await cache_service.get("absent") is None

    async def test_tracks_hits_and_misses(self, cache_service: CacheService) -> None:
        await cache_service.set("k", {"v": 1})
        await cache_service.get("k")
        await cache_service.get("absent")

        assert cache_service.hits == 1
        assert cache_service.misses == 1
        assert cache_service.hit_rate == 0.5

    async def test_delete_removes_an_entry(self, cache_service: CacheService) -> None:
        await cache_service.set("k", {"v": 1})
        await cache_service.delete("k")
        assert await cache_service.get("k") is None

    async def test_ping_reports_connectivity(self, cache_service: CacheService) -> None:
        healthy, detail = await cache_service.ping()
        assert healthy
        assert detail is None


class TestFailsOpen:
    """A cache problem must degrade to uncached inference, never an error."""

    async def test_disabled_cache_is_inert(self) -> None:
        cache = CacheService(None)

        assert not cache.enabled
        assert await cache.get("k") is None
        assert await cache.set("k", {"v": 1}) is False
        await cache.delete("k")  # must not raise

    async def test_get_swallows_redis_errors(
        self, cache_service: CacheService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _fail(*args: object, **kwargs: object) -> None:
            raise RedisConnectionError("connection lost")

        monkeypatch.setattr(cache_service._redis, "get", _fail)

        assert await cache_service.get("k") is None
        assert cache_service.errors == 1

    async def test_set_swallows_redis_errors(
        self, cache_service: CacheService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _fail(*args: object, **kwargs: object) -> None:
            raise RedisConnectionError("connection lost")

        monkeypatch.setattr(cache_service._redis, "set", _fail)

        assert await cache_service.set("k", {"v": 1}) is False
        assert cache_service.errors == 1

    async def test_corrupt_entry_is_discarded_not_returned(
        self, cache_service: CacheService
    ) -> None:
        """A malformed entry must be evicted and treated as a miss.

        Returning it would propagate corruption into a response; leaving it in
        place would make every subsequent request for that key fail.
        """
        assert cache_service._redis is not None
        await cache_service._redis.set("k", "{not valid json")

        assert await cache_service.get("k") is None
        assert cache_service.errors == 1
        assert await cache_service._redis.get("k") is None

    async def test_unserialisable_value_does_not_raise(self, cache_service: CacheService) -> None:
        """A value that cannot be serialised is dropped, not propagated.

        A self-referential structure is used rather than a bare object: the
        serialiser is configured with ``default=str``, which successfully
        stringifies most unexpected values. A cycle is one of the few inputs
        that genuinely fails, and it must not escape into the request path.
        """
        cyclic: dict[str, object] = {}
        cyclic["self"] = cyclic

        assert await cache_service.set("k", cyclic) is False
        assert cache_service.errors == 1

    async def test_ping_reports_failure_without_raising(self) -> None:
        cache = CacheService(None)
        healthy, detail = await cache.ping()
        assert not healthy
        assert detail


class TestStats:
    async def test_reports_operational_counters(self, cache_service: CacheService) -> None:
        await cache_service.set("k", {"v": 1})
        await cache_service.get("k")
        await cache_service.get("absent")

        stats = cache_service.stats()
        assert stats["enabled"] is True
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    async def test_hit_rate_is_zero_with_no_traffic(self) -> None:
        assert CacheService(None).hit_rate == 0.0
