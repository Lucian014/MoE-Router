from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Shared declarative base. Every table's schema.py imports THIS Base
    """

    pass
