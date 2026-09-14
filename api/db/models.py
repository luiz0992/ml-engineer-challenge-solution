"""Database models.

Currently one table: an append-only record of every inference the service
performs. It exists to answer questions that metrics cannot, because Prometheus
aggregates and discards individual events:

* Which model version produced the prediction a customer is disputing?
* What was the input distribution last Tuesday, for drift analysis?
* Which user is responsible for the latency spike at 14:05?

Design decisions that matter at volume:

**No image bytes are stored.** Only a SHA-256 digest of the upload. Storing
images would grow the table without bound, and would put user-supplied content
into a database that has a different retention and access-control story than
the object store where such content belongs.

**Indexes are chosen for the queries above**, not added per column. Every index
costs write throughput on an append-heavy table, and this table is written once
per inference.

**Timestamps are timezone-aware.** A naive timestamp in a distributed system is
ambiguous the moment a container runs in a different zone than the database.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all models."""


class InferenceLog(Base):
    """One inference request, successful or not.

    Failures are recorded as well as successes: an error rate computed only
    from successful rows is meaningless, and the failures are usually what an
    investigation is about.
    """

    __tablename__ = "inference_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # --- Request identity -------------------------------------------------
    #: Ties this row to the application logs for the same request.
    correlation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    user_tier: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # --- Model provenance -------------------------------------------------
    # Denormalised deliberately. A foreign key to a models table would make
    # history mutable: renaming or deleting a model version would rewrite what
    # past predictions claim to have used. An audit record must stay true.
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    backend: Mapped[str] = mapped_column(String(32), nullable=False)
    task: Mapped[str] = mapped_column(String(32), nullable=False)

    #: A/B experiment arm, when one applied. Null for the common case of no
    #: active experiment. Stored as a column rather than in `notes` because the
    #: comparison in models/validation/ab_testing.py groups by it, and grouping
    #: on a parsed free-text field would be both slow and fragile.
    variant: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # --- Outcome ----------------------------------------------------------
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)

    #: Server-side duration: preprocessing plus the forward pass.
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    cached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # --- Input fingerprint -------------------------------------------------
    #: SHA-256 of the upload. Identifies repeat submissions and links a dispute
    #: to a specific input without retaining the image itself.
    image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_format: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # --- Prediction --------------------------------------------------------
    # Only the top prediction. The full ranking is reconstructible by replaying
    # the input against the recorded model version, and storing every class for
    # every request would multiply the table size for data that is almost never
    # read.
    top_label: Mapped[str | None] = mapped_column(String(256), nullable=True)
    top_class_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    top_probability: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- Timing ------------------------------------------------------------
    # server_default=now() so a row inserted by any client, including psql
    # during an incident, is timestamped by the database rather than by a
    # possibly-skewed application clock.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(UTC),
    )

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # Time-ordered scans: dashboards and retention jobs both read recent
        # rows first. Descending matches the query direction, so Postgres can
        # walk the index without a sort.
        Index("ix_inference_logs_created_at", created_at.desc()),
        # Per-model analysis over a window: the drift and A/B comparisons in
        # models/validation/ use exactly this shape.
        #
        # This index only earns its keep once several model versions coexist,
        # which is what the versioning and A/B features produce. With a single
        # deployed model the planner correctly prefers a scan filtered by
        # created_at, and this becomes the largest index on the table for no
        # benefit. Review it if the service only ever runs one model.
        Index("ix_inference_logs_model_created", model_name, model_version, created_at.desc()),
        # Per-user investigation and quota reconciliation.
        Index("ix_inference_logs_user_created", user_id, created_at.desc()),
        # Error triage. Partial, so it indexes only the rows an investigation
        # reads and costs nothing on the successful majority.
        Index(
            "ix_inference_logs_errors",
            created_at.desc(),
            postgresql_where=status != "success",
        ),
        # Repeat-submission analysis and cache effectiveness.
        Index("ix_inference_logs_image_sha256", image_sha256),
        # A/B comparison reads one experiment's arms over a window. Partial, so
        # it costs nothing on the overwhelming majority of rows that belong to
        # no experiment.
        Index(
            "ix_inference_logs_variant",
            variant,
            created_at.desc(),
            postgresql_where=variant.is_not(None),
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<InferenceLog id={self.id} {self.model_name}:{self.model_version} "
            f"status={self.status} latency={self.latency_ms:.1f}ms>"
        )
