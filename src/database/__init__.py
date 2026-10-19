from database.base import Base

# Re-export every table's public pieces here too, so callers can do
# `from database import ModelRepository` instead of reaching into
# `database.models.model_registry`. Add one line per new table.
from database.models.model_registry import (  # noqa: F401
    ModelAlreadyExistsError,
    ModelNotFoundError,
    ModelRegistryEntry,
    ModelRepository,
)
from database.session import dispose_engine, get_engine, get_session

__all__ = [
    "Base",
    "dispose_engine",
    "get_engine",
    "get_session",
    "ModelRegistryEntry",
    "ModelRepository",
    "ModelAlreadyExistsError",
    "ModelNotFoundError",
]
