# Every table lives in its own folder under database/models/<table_name>/,
# each with a schema.py (ORM class) and a repository.py (CRUD/queries).
#
# IMPORTANT: every schema module must be imported here. Alembic's
# autogenerate compares the live database against `Base.metadata`
# (database/base.py), and a schema class only registers itself onto that
# metadata the moment its module is imported. Forgetting to add a new
# table's import here means `alembic revision --autogenerate` will silently
# not see it.
#
# When you add a new table folder, add one line here:
#   from database.models.<table_name>.schema import <ORMClassName>  # noqa: F401

from database.models.chat_session.schema import ChatSession  # noqa: F401
from database.models.chunk.schema import Chunk  # noqa: F401
from database.models.model_registry.schema import ModelRegistryEntry  # noqa: F401
from database.models.turn.schema import Turn  # noqa: F401
