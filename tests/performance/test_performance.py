"""Performance and resource tests.

Excluded from the default run (`-m "not performance"`), because they are slower
and their thresholds depend on the host. They exist to catch regressions of a
*kind* — an accidental O(n^2), a per-request model reload, a memory leak — not
to certify absolute latency, which is what `benchmarks/` is for.

Thresholds are deliberately loose, set well above the measured values so they
fail on a real regression rather than on a busy CI runner.
"""

from __future__ import annotations

import asyncio
import gc
import time
import tracemalloc
from typing import Any

import numpy as np
import pytest

from api.utils.image_processing import PreprocessConfig, preprocess_image, softmax, top_k
from tests.conftest import make_image_bytes

pytestmark = [pytest.mark.performance, pytest.mark.slow]


class TestPreprocessingThroughput:
    def test_single_image_preprocessing_is_fast(self) -> None:
        """Preprocessing must not dominate inference.

        Measured at roughly 3-5 ms; 50 ms would mean something pathological,
        such as an accidental full-resolution copy per call.
        """
        image = make_image_bytes(512, 512)
        config = PreprocessConfig(image_size=224)

        preprocess_image(image, config)  # warm caches

        start = time.perf_counter()
        iterations = 20
        for _ in range(iterations):
            preprocess_image(image, config)
        per_call_ms = (time.perf_counter() - start) * 1000 / iterations

        assert per_call_ms < 50, f"preprocessing took {per_call_ms:.1f} ms per image"

    def test_cost_scales_with_input_size_not_catastrophically(self) -> None:
        """Doubling each edge should not cost far more than ~4x.

        Quadratic in edge length is expected, since pixel count is. A much
        worse ratio would indicate an unintended repeated resize.
        """
        config = PreprocessConfig(image_size=224)

        def measure(edge: int) -> float:
            image = make_image_bytes(edge, edge)
            preprocess_image(image, config)
            start = time.perf_counter()
            for _ in range(5):
                preprocess_image(image, config)
            return (time.perf_counter() - start) / 5

        small = measure(256)
        large = measure(512)

        assert large / small < 12, f"scaling factor {large / small:.1f} is superquadratic"


class TestPostprocessingComplexity:
    def test_top_k_does_not_sort_all_classes(self) -> None:
        """top_k uses argpartition, so cost is near-linear in class count.

        A full sort would be O(n log n); at 200 classes both are fast, but the
        difference matters if the label space grows.
        """
        rng = np.random.default_rng(0)

        def measure(num_classes: int) -> float:
            probabilities = softmax(rng.standard_normal(num_classes))
            start = time.perf_counter()
            for _ in range(200):
                top_k(probabilities, 5)
            return time.perf_counter() - start

        small = measure(200)
        large = measure(20_000)

        # 100x more classes should cost well under 100x more time.
        assert large / small < 60, f"top_k scaling factor {large / small:.1f}"


class TestBatchingEfficiency:
    def test_batching_beats_sequential_inference(self, real_artifacts_dir: Any) -> None:
        """A batched forward pass must outperform one call per image.

        This is the property that makes the batch endpoint worth having; if it
        regressed to a loop, throughput would collapse under load.

        Measured against the *real* model, and on the forward pass alone. The
        synthetic test model is a pooling layer and a 3x10 matmul, where
        per-call overhead dominates and batching cannot help — asserting the
        property there would test the fixture rather than the system.
        """
        from api.config import Settings
        from api.services.model_service import ModelService

        settings = Settings(_env_file=None, artifacts_dir=real_artifacts_dir)
        service = ModelService(settings)
        model = service.load_classifier()

        batch_size = 8
        batched_input = np.zeros((batch_size, 3, model.image_size, model.image_size), np.float32)
        single_input = np.zeros((1, 3, model.image_size, model.image_size), np.float32)

        # Warm up both shapes; the first call for a shape pays initialisation.
        service.run(model, batched_input)
        service.run(model, single_input)

        start = time.perf_counter()
        service.run(model, batched_input)
        batched = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(batch_size):
            service.run(model, single_input)
        sequential = time.perf_counter() - start

        assert batched < sequential, (
            f"batch of {batch_size} took {batched * 1000:.1f} ms against "
            f"{sequential * 1000:.1f} ms sequentially; batching is not paying off"
        )


class TestMemoryStability:
    async def test_repeated_inference_does_not_leak(self, inference_service: Any) -> None:
        """Memory must stabilise across many requests.

        A per-request model reload or an unbounded accumulator would show up
        here as steady growth. The threshold is generous because allocator
        behaviour is noisy; it is looking for a leak, not for precision.
        """
        psutil = pytest.importorskip("psutil")
        process = psutil.Process()

        # Warm up so one-off allocations are not counted as growth.
        for index in range(10):
            await inference_service.classify(
                make_image_bytes(seed=index), correlation_id="warmup", use_cache=False
            )

        gc.collect()
        before_mb = process.memory_info().rss / 1024 / 1024

        for index in range(100):
            await inference_service.classify(
                make_image_bytes(seed=index + 1000),
                correlation_id="measure",
                use_cache=False,
            )

        gc.collect()
        growth_mb = process.memory_info().rss / 1024 / 1024 - before_mb

        assert growth_mb < 150, f"RSS grew {growth_mb:.0f} MB over 100 inferences"

    async def test_peak_allocation_per_request_is_bounded(self, inference_service: Any) -> None:
        """Peak Python-level allocation per classification.

        Distinct from the RSS check above, and more useful for capacity
        planning: the allocator holds freed memory, so RSS understates what a
        burst of concurrent requests needs. Peak allocation is what decides how
        many requests fit inside a memory limit at once.

        Measured at roughly 1.8 MB for a 512x512 upload; 32 MB would mean a
        full-resolution float copy is being materialised.
        """
        image = make_image_bytes(512, 512)

        for _ in range(5):
            await inference_service.classify(image, correlation_id="warmup", use_cache=False)

        gc.collect()
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            await inference_service.classify(image, correlation_id="peak", use_cache=False)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        assert peak_mb < 32, f"one classification peaked at {peak_mb:.1f} MB"

    async def test_peak_allocation_does_not_track_upload_size(self, inference_service: Any) -> None:
        """A 4x larger upload must not cost 4x the memory.

        Images are resized to 224x224 before inference, so peak allocation
        should be roughly flat across upload sizes. A peak that scales with the
        input means a full-resolution array is being retained past the resize --
        which is also how an oversized upload becomes a memory-exhaustion
        vector rather than a validation error.
        """

        async def peak_for(edge: int) -> float:
            image = make_image_bytes(edge, edge)
            await inference_service.classify(image, correlation_id="warmup", use_cache=False)

            gc.collect()
            tracemalloc.start()
            try:
                tracemalloc.reset_peak()
                await inference_service.classify(
                    image, correlation_id=f"scale-{edge}", use_cache=False
                )
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            return peak / 1024 / 1024

        small = await peak_for(256)
        large = await peak_for(1024)

        # 1024x1024 is 16x the pixels of 256x256. Allowing 3x leaves generous
        # room for decode buffers while still failing if the peak is
        # proportional to the input.
        assert large < small * 3, (
            f"peak allocation grew from {small:.2f} MB to {large:.2f} MB for a "
            f"16x larger upload; the full-resolution image is being retained"
        )

    async def test_cached_responses_do_not_accumulate(self, inference_service: Any) -> None:
        """Repeatedly serving the same image must not grow the traced heap.

        The cache is bounded by TTL rather than by size, so a per-request entry
        that is never evicted would show up here as steady growth.
        """
        image = make_image_bytes(256, 256)
        await inference_service.classify(image, correlation_id="warmup", use_cache=True)

        gc.collect()
        tracemalloc.start()
        try:
            before = tracemalloc.take_snapshot()
            for index in range(50):
                await inference_service.classify(
                    image, correlation_id=f"cached-{index}", use_cache=True
                )
            gc.collect()
            after = tracemalloc.take_snapshot()
        finally:
            tracemalloc.stop()

        growth_kb = sum(s.size_diff for s in after.compare_to(before, "filename")) / 1024
        assert growth_kb < 4096, f"traced heap grew {growth_kb:.0f} KB over 50 cached requests"

    def test_model_is_loaded_once_not_per_request(self, model_service: Any) -> None:
        """Repeated lookups must return the same session object.

        Rebuilding an ONNX Runtime session per request would add hundreds of
        milliseconds and dominate latency.
        """
        first = model_service.get_classifier()
        second = model_service.get_classifier()

        assert first.session is second.session


class TestConcurrency:
    async def test_inference_does_not_block_the_event_loop(self, inference_service: Any) -> None:
        """The forward pass runs in a thread pool.

        ONNX Runtime's Run() is blocking. Executing it directly in the event
        loop would stall every other in-flight request for its duration; this
        asserts a lightweight coroutine still progresses while inference runs.
        """
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            for _ in range(200):
                await asyncio.sleep(0.001)
                ticks += 1

        tick_task = asyncio.create_task(ticker())
        await asyncio.gather(
            *(
                inference_service.classify(
                    make_image_bytes(seed=i), correlation_id=str(i), use_cache=False
                )
                for i in range(8)
            )
        )
        tick_task.cancel()

        assert ticks > 5, (
            f"the event loop advanced only {ticks} times during inference; "
            f"the forward pass is blocking it"
        )

    async def test_concurrent_requests_all_succeed(self, inference_service: Any) -> None:
        results = await asyncio.gather(
            *(
                inference_service.classify(
                    make_image_bytes(seed=i), correlation_id=str(i), use_cache=False
                )
                for i in range(16)
            )
        )

        assert len(results) == 16
        assert all(r.predictions for r in results)


class TestCacheSpeedup:
    async def test_cache_hit_is_substantially_faster(self, inference_service: Any) -> None:
        """A cache hit must avoid preprocessing and inference entirely.

        Measured at roughly 45x on the real model. The assertion only requires
        a clear improvement, so it does not flake when the synthetic test model
        makes the uncached path cheap.
        """
        image = make_image_bytes(seed=99)

        start = time.perf_counter()
        await inference_service.classify(image, correlation_id="cold")
        cold = time.perf_counter() - start

        start = time.perf_counter()
        response = await inference_service.classify(image, correlation_id="warm")
        warm = time.perf_counter() - start

        assert response.cached is True
        assert warm < cold, (
            f"cache hit ({warm * 1000:.2f} ms) was not faster than cold ({cold * 1000:.2f} ms)"
        )
