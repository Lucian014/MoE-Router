import datetime

from sqlalchemy import ARRAY, Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class ModelRegistryEntry(Base):
    """
    A single routable model, replacing one entry of the old
    router/models.json array.

    `model_id` is the natural key (e.g. "qwen2.5-coder:7b") — it's what the
    semantic router currently keeps in _MODEL_NAMES and what
    shared.utils.model_to_queue() maps to a RabbitMQ routing key, so it must
    stay unique and stable.
    """

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    model_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    engine: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Postgres ARRAY(Text) mirrors the `examples: list[str]` field in the old
    # YAML/JSON catalog. If cross-database portability is ever needed (e.g.
    # SQLite in tests), swap this for a JSON column instead.
    examples: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, server_default="{}"
    )

    replicas: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelRegistryEntry model_id={self.model_id!r} engine={self.engine!r} "
            f"enabled={self.enabled} replicas={self.replicas}>"
        )

    def to_dict(self) -> dict:
        """Shape compatible with the old models.json entries, for an easy
        drop-in once router.py is migrated to read from Postgres."""
        return {
            "model_id": self.model_id,
            "engine": self.engine,
            "description": self.description,
            "examples": list(self.examples or []),
            "replicas": self.replicas,
            "enabled": self.enabled,
        }
