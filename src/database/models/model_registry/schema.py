import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, Index, Integer, String, func, text
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base

COMPLEXITY_TIERS = ("low", "high")


class ModelRegistryEntry(Base):
    __tablename__ = "models"
    __table_args__ = (
        CheckConstraint("complexity_tier IN ('low', 'high')", name="ck_models_complexity_tier"),
        Index(
            "ux_models_category_tier", "category", "complexity_tier", unique=True,
            postgresql_where=text("enabled AND category IS NOT NULL"),
        ),
        Index(
            "ux_models_single_default", "is_default", unique=True,
            postgresql_where=text("is_default"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    model_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    engine: Mapped[str] = mapped_column(String(64), nullable=False)

    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    complexity_tier: Mapped[str] = mapped_column(
        String(16), nullable=False, default="high", server_default="high"
    )

    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")

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
            f"category={self.category!r} tier={self.complexity_tier!r} enabled={self.enabled} "
            f"is_default={self.is_default}>"
        )

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "engine": self.engine,
            "category": self.category,
            "complexity_tier": self.complexity_tier,
            "enabled": self.enabled,
            "is_default": self.is_default,
        }