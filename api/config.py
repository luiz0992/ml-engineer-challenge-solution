"""Application configuration.

Settings are loaded from environment variables (and a local ``.env`` file
during development) into a typed, validated Pydantic model.

Two properties of this module matter for production readiness:

1. **No secret has a usable default.** ``JWT_SECRET_KEY`` ships as an obvious
   placeholder and :meth:`Settings._reject_placeholder_secret` refuses to boot
   a production process that still carries it. A misconfigured deploy fails
   loudly at startup rather than silently authenticating with a known key.
2. **Settings are read once.** :func:`get_settings` is cached, so the object is
   built a single time per process and injected everywhere via FastAPI's
   dependency system, which also makes it trivial to override in tests.
"""

from __future__ import annotations

import secrets
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_SECRET = "CHANGE_ME_generate_with_openssl_rand_hex_32"  # noqa: S105 - template sentinel

#: Development-only credential defaults. Permitted outside production so the
#: stack starts with no configuration; rejected in production by
#: :meth:`Settings._reject_insecure_defaults`.
DEV_DEFAULT_DB_PASSWORD = "mlapi"  # noqa: S105 - non-secret dev placeholder


class AppEnv(StrEnum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class UserTier(StrEnum):
    """Subscription tier, which determines a caller's rate limit."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class InferenceBackend(StrEnum):
    """Runtime used to execute a model.

    Ordered from most portable to most specialised. ``TENSORRT`` is the
    fastest but requires an NVIDIA GPU and an engine built for that exact
    GPU architecture, so it is never the default.
    """

    TORCH = "torch"
    ONNX = "onnx"
    ONNX_INT8 = "onnx-int8"
    TENSORRT = "tensorrt"


class Settings(BaseSettings):
    """Typed application settings, populated from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ------------------------------------------------------
    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    api_v1_prefix: str = "/api/v1"

    # --- Security ---------------------------------------------------------
    jwt_secret_key: str = PLACEHOLDER_SECRET
    jwt_algorithm: str = "HS256"
    jwt_access_token_ttl_seconds: int = 3600

    # --- Postgres ---------------------------------------------------------
    postgres_user: str = "mlapi"
    postgres_password: str = DEV_DEFAULT_DB_PASSWORD
    postgres_db: str = "mlapi"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # --- Redis ------------------------------------------------------------
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl_seconds: int = 3600

    # --- Celery -----------------------------------------------------------
    # Typed as `str` rather than `RedisDsn` because Celery's constructor takes
    # a plain string; carrying a Pydantic URL type here would only force a
    # conversion at every use site.
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    # --- Model serving ----------------------------------------------------
    artifacts_dir: Path = Path("models/artifacts")
    inference_backend: InferenceBackend = InferenceBackend.ONNX
    enable_graceful_degradation: bool = True
    #: Allow `?backend=` on inference endpoints. Off by default so a caller
    #: cannot force INT8 in production without an operator flipping this.
    allow_backend_override: bool = False

    # --- Request limits ---------------------------------------------------
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 89_478_485
    max_batch_size: int = 64

    # --- Rate limiting ----------------------------------------------------
    rate_limit_free_rpm: int = 10
    rate_limit_pro_rpm: int = 120
    rate_limit_enterprise_rpm: int = 1200

    # --- Derived ----------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.app_env is AppEnv.PRODUCTION

    @property
    def database_url(self) -> str:
        """Async SQLAlchemy DSN, used by the API process."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Synchronous DSN, used by Alembic migrations and Celery workers."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    def rate_limit_for(self, tier: UserTier) -> int:
        """Requests per minute permitted for ``tier``."""
        return {
            UserTier.FREE: self.rate_limit_free_rpm,
            UserTier.PRO: self.rate_limit_pro_rpm,
            UserTier.ENTERPRISE: self.rate_limit_enterprise_rpm,
        }[tier]

    # --- Validation -------------------------------------------------------
    @model_validator(mode="after")
    def _reject_insecure_defaults(self) -> Self:
        """Refuse to run in production with template credentials.

        Outside production we substitute a random per-process signing key
        instead of raising, so developers and CI can start the app with no
        setup while still never sharing a hardcoded secret. The trade-off is
        that tokens do not survive a restart locally, which is harmless.

        In production both the signing key and the database password must be
        supplied explicitly. Failing at startup turns a silent security hole
        into an obvious, immediate deployment failure.
        """
        if self.is_production:
            insecure: list[str] = []
            if self.jwt_secret_key == PLACEHOLDER_SECRET:
                insecure.append("JWT_SECRET_KEY (generate with `openssl rand -hex 32`)")
            if self.postgres_password == DEV_DEFAULT_DB_PASSWORD:
                insecure.append("POSTGRES_PASSWORD (still the development default)")
            if insecure:
                raise ValueError(
                    "Refusing to start in production with insecure defaults: "
                    + "; ".join(insecure)
                    + ". Inject real values via the secret store."
                )
        elif self.jwt_secret_key == PLACEHOLDER_SECRET:
            object.__setattr__(self, "jwt_secret_key", secrets.token_hex(32))
        return self

    @model_validator(mode="after")
    def _validate_limits(self) -> Self:
        if self.max_batch_size < 1:
            raise ValueError("MAX_BATCH_SIZE must be at least 1")
        if self.max_upload_bytes < 1024:
            raise ValueError("MAX_UPLOAD_BYTES must be at least 1 KiB")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that the environment is parsed once. Tests clear the cache via
    ``get_settings.cache_clear()`` to inject overrides.
    """
    return Settings()
