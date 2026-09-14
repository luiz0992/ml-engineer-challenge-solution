"""Tests for the asynchronous batch endpoints.

Celery is stubbed rather than run: these tests verify the HTTP contract —
validation, the 202 acknowledgement, status mapping, and the distinction
between an unknown job and a pending one — not Celery's own correctness, which
is its maintainers' responsibility.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.integration

BATCH = "/api/v1/batch"


class _StubAsyncResult:
    """Stands in for a Celery task handle."""

    def __init__(self, task_id: str = "job-123") -> None:
        self.id = task_id


@pytest.fixture
def submitted_jobs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture submissions instead of enqueuing them."""
    captured: list[dict[str, Any]] = []

    def _delay(items: list[dict[str, Any]], **kwargs: Any) -> _StubAsyncResult:
        captured.append({"items": items, **kwargs})
        return _StubAsyncResult()

    import worker.tasks

    monkeypatch.setattr(worker.tasks.process_batch, "delay", _delay)
    return captured


class TestSubmission:
    async def test_accepts_a_batch_and_returns_a_job_handle(
        self, client: Any, auth_headers: dict[str, str], submitted_jobs: list[dict[str, Any]]
    ) -> None:
        response = await client.post(
            BATCH,
            json={
                "task": "classification",
                "items": [
                    {"image_url": "https://example.com/a.jpg", "item_id": "a"},
                    {"image_url": "https://example.com/b.jpg"},
                ],
            },
            headers=auth_headers,
        )
        body = response.json()

        # 202, not 200: the work has been accepted, not completed.
        assert response.status_code == 202
        assert body["status"] == "pending"
        assert body["total_items"] == 2
        assert body["job_id"]
        # The client is told where to poll rather than having to construct it.
        assert body["job_id"] in body["status_url"]

        assert len(submitted_jobs) == 1
        assert submitted_jobs[0]["items"][0]["item_id"] == "a"

    async def test_requires_authentication(self, client: Any) -> None:
        response = await client.post(
            BATCH, json={"items": [{"image_url": "https://example.com/a.jpg"}]}
        )
        assert response.status_code == 401

    async def test_rejects_an_empty_batch(self, client: Any, auth_headers: dict[str, str]) -> None:
        response = await client.post(BATCH, json={"items": []}, headers=auth_headers)
        assert response.status_code == 422

    async def test_rejects_non_http_image_urls(
        self, client: Any, auth_headers: dict[str, str]
    ) -> None:
        """SSRF protection is enforced at the endpoint, not only in the worker.

        Rejecting at submission means a hostile URL never reaches a worker at
        all.
        """
        response = await client.post(
            BATCH,
            json={"items": [{"image_url": "file:///etc/passwd"}]},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_rejects_an_oversized_batch(
        self, client: Any, auth_headers: dict[str, str], settings: Any
    ) -> None:
        """Rejected before enqueueing, so no worker is occupied by a doomed job."""
        too_many = settings.max_batch_size * 10 + 1
        response = await client.post(
            BATCH,
            json={"items": [{"image_url": f"https://e.com/{i}.jpg"} for i in range(too_many)]},
            headers=auth_headers,
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "batch_too_large"

    async def test_forwards_classification_options_to_the_worker(
        self, client: Any, auth_headers: dict[str, str], submitted_jobs: list[dict[str, Any]]
    ) -> None:
        await client.post(
            BATCH,
            json={
                "items": [{"image_url": "https://example.com/a.jpg"}],
                "classification_options": {"top_k": 3},
            },
            headers=auth_headers,
        )
        assert submitted_jobs[0]["options"]["top_k"] == 3


class TestStatus:
    async def test_unknown_job_returns_404_not_pending(
        self, client: Any, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Celery reports an unknown ID as PENDING.

        It cannot distinguish a job that has not started from one that never
        existed. Passing that through would leave a client polling forever for
        a typo, so a PENDING job with no backend record is reported as absent.
        """

        class _Unknown:
            state = "PENDING"
            info: dict[str, Any] = {}

            class backend:  # noqa: N801
                @staticmethod
                def get_task_meta(_: str) -> dict[str, Any]:
                    return {}

        monkeypatch.setattr("api.routers.batch.AsyncResult", lambda _: _Unknown())

        response = await client.get(f"{BATCH}/no-such-job", headers=auth_headers)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "job_not_found"

    async def test_reports_progress_while_running(
        self, client: Any, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Running:
            state = "PROGRESS"
            info = {"total_items": 10, "completed_items": 4, "failed_items": 1}

            class backend:  # noqa: N801
                @staticmethod
                def get_task_meta(_: str) -> dict[str, Any]:
                    return {"status": "PROGRESS"}

        monkeypatch.setattr("api.routers.batch.AsyncResult", lambda _: _Running())

        body = (await client.get(f"{BATCH}/job-1", headers=auth_headers)).json()

        assert body["status"] == "running"
        assert body["completed_items"] == 4
        assert body["failed_items"] == 1

    async def test_returns_per_item_results_when_complete(
        self, client: Any, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed item must not discard the successful ones."""

        class _Complete:
            state = "SUCCESS"
            info: dict[str, Any] = {}
            result = {
                "task": "classification",
                "total_items": 2,
                "completed_items": 1,
                "failed_items": 1,
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:00:05+00:00",
                "results": [
                    {
                        "item_id": "a",
                        "image_url": "https://e.com/a.jpg",
                        "status": "succeeded",
                        "result": {"predictions": [{"label": "cat"}]},
                        "error": None,
                    },
                    {
                        "item_id": "b",
                        "image_url": "https://e.com/b.jpg",
                        "status": "failed",
                        "result": None,
                        "error": {"code": "fetch_failed", "message": "unreachable"},
                    },
                ],
            }

            class backend:  # noqa: N801
                @staticmethod
                def get_task_meta(_: str) -> dict[str, Any]:
                    return {"status": "SUCCESS"}

        monkeypatch.setattr("api.routers.batch.AsyncResult", lambda _: _Complete())

        body = (await client.get(f"{BATCH}/job-1", headers=auth_headers)).json()

        assert body["status"] == "completed"
        assert len(body["results"]) == 2
        assert body["results"][0]["status"] == "succeeded"
        assert body["results"][1]["error"]["code"] == "fetch_failed"

    async def test_failure_does_not_leak_internal_detail(
        self, client: Any, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exception text stays in the worker logs.

        A traceback can carry file paths and configuration; the client gets the
        job ID to quote instead.
        """

        class _Failed:
            state = "FAILURE"
            info = {"total_items": 1}
            result = RuntimeError("/app/secret/path exploded: password=hunter2")

            class backend:  # noqa: N801
                @staticmethod
                def get_task_meta(_: str) -> dict[str, Any]:
                    return {"status": "FAILURE"}

        monkeypatch.setattr("api.routers.batch.AsyncResult", lambda _: _Failed())

        body = (await client.get(f"{BATCH}/job-1", headers=auth_headers)).json()

        assert body["status"] == "failed"
        assert "hunter2" not in body["error"]
        assert "/app/secret" not in body["error"]

    async def test_requires_authentication(self, client: Any) -> None:
        assert (await client.get(f"{BATCH}/job-1")).status_code == 401
