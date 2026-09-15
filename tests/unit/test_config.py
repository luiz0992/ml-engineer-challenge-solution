"""Tests for application configuration.

The production guard is the important behaviour here: a deploy carrying a
template credential must fail at startup rather than authenticate with a key
that is published in the repository.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.config import (
    DEV_DEFAULT_DB_PASSWORD,
    PLACEHOLDER_SECRET,
    AppEnv,
    InferenceBackend,
    Settings,
    UserTier,
    get_settings,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build every `Settings` in this module from defaults alone.

    `_env_file=None` keeps a developer's .env out of the run, but not the
    ambient environment -- pydantic-settings still reads real variables. CI
    exports `POSTGRES_PASSWORD` and `JWT_SECRET_KEY` for its Postgres service
    and application config, which silently satisfied the insecure-default
    assertions: three tests here passed on a laptop and failed on the runner.

    Cleared by field name, so a new setting cannot quietly reintroduce the
    dependency.
    """
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
        monkeypatch.delenv(field.lower(), raising=False)


class TestProductionGuard:
    def test_development_generates_a_random_key(self) -> None:
        """Local runs need no setup but still never share a hardcoded secret.

        The trade-off is that tokens do not survive a restart locally, which is
        harmless.
        """
        settings = Settings(_env_file=None)

        assert settings.jwt_secret_key != PLACEHOLDER_SECRET
        assert len(settings.jwt_secret_key) == 64

    def test_development_keys_are_unique_per_process(self) -> None:
        first = Settings(_env_file=None).jwt_secret_key
        second = Settings(_env_file=None).jwt_secret_key
        assert first != second

    def test_production_rejects_the_template_signing_key(self) -> None:
        with pytest.raises(ValidationError, match="JWT_SECRET_KEY"):
            Settings(
                _env_file=None,
                app_env=AppEnv.PRODUCTION,
                postgres_password="a-real-password",
            )

    def test_production_rejects_the_development_database_password(self) -> None:
        with pytest.raises(ValidationError, match="POSTGRES_PASSWORD"):
            Settings(
                _env_file=None,
                app_env=AppEnv.PRODUCTION,
                jwt_secret_key="a" * 64,
                postgres_password=DEV_DEFAULT_DB_PASSWORD,
            )

    def test_production_reports_every_insecure_default_at_once(self) -> None:
        """Both problems are named together, so one deploy fixes both."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, app_env=AppEnv.PRODUCTION)

        message = str(exc_info.value)
        assert "JWT_SECRET_KEY" in message
        assert "POSTGRES_PASSWORD" in message

    def test_production_starts_with_real_credentials(self) -> None:
        settings = Settings(
            _env_file=None,
            app_env=AppEnv.PRODUCTION,
            jwt_secret_key="a" * 64,
            postgres_password="a-real-password",
        )
        assert settings.is_production


class TestDerivedValues:
    def test_async_and_sync_dsns_use_different_drivers(self) -> None:
        """The API uses asyncpg; Alembic and Celery need a sync driver."""
        settings = Settings(_env_file=None)

        assert settings.database_url.startswith("postgresql+asyncpg://")
        assert settings.sync_database_url.startswith("postgresql+psycopg://")

    def test_dsns_include_the_configured_target(self) -> None:
        settings = Settings(
            _env_file=None, postgres_host="db.internal", postgres_port=5433, postgres_db="prod"
        )
        assert "db.internal:5433/prod" in settings.database_url

    def test_redis_url_reflects_configuration(self) -> None:
        settings = Settings(_env_file=None, redis_host="cache", redis_port=6380, redis_db=3)
        assert settings.redis_url == "redis://cache:6380/3"

    @pytest.mark.parametrize(
        ("tier", "expected"),
        [(UserTier.FREE, 10), (UserTier.PRO, 120), (UserTier.ENTERPRISE, 1200)],
    )
    def test_rate_limits_increase_with_tier(self, tier: UserTier, expected: int) -> None:
        assert Settings(_env_file=None).rate_limit_for(tier) == expected


class TestValidation:
    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_rejects_invalid_batch_size(self, batch_size: int) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, max_batch_size=batch_size)

    def test_rejects_implausibly_small_upload_limit(self) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, max_upload_bytes=10)

    def test_rejects_unknown_inference_backend(self) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, inference_backend="quantum")

    def test_accepts_every_declared_backend(self) -> None:
        for backend in InferenceBackend:
            assert Settings(_env_file=None, inference_backend=backend).inference_backend is backend


class TestSettingsCache:
    def test_settings_are_constructed_once(self) -> None:
        """The environment is parsed once per process, not per request."""
        get_settings.cache_clear()
        try:
            assert get_settings() is get_settings()
        finally:
            get_settings.cache_clear()
