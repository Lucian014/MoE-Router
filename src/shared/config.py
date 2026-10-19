"""
shared/config.py -- the single source of truth for every runtime variable in src/.

Each service reads its settings from here instead of calling os.getenv() itself:

    from shared.config import get_rabbitmq_settings
    rabbit = get_rabbitmq_settings()        # validated on first call, then cached

Values come from environment variables (Kubernetes injects them from the manifests /
Secrets) or, for local runs, from a `.env` file in the working directory (see .env.example).
Real environment variables win over `.env`.

The getters are lazy on purpose: nothing is validated at import time, so a module can be
imported without any env (the router image runs `import router.router` at build time), and
each service only validates the settings it actually uses (the gateway never needs MODEL_NAME).

Variable                 Used by              Default
-----------------------  -------------------  ------------------------------------------
POSTGRES_HOST            router, tools        postgres
POSTGRES_PORT                                 5432
POSTGRES_DB                                   model_registry
POSTGRES_USER                                 (required)
POSTGRES_PASSWORD                             (required)
RABBITMQ_HOST            gateway, router,     rabbitmq
RABBITMQ_PORT            worker               5672
RABBITMQ_USER                                 (required)
RABBITMQ_PASS                                 (required)
MIDDLEWARE_QUEUE         gateway, router      router.requests
CATALOG_REFRESH_SECONDS  router               30
DOMAIN_CHECKPOINT        router (classifier)  router/checkpoints/laya-domain-v4, else HF repo
COMPLEXITY_CHECKPOINT                         router/checkpoints/laya-complexity-v2, else HF repo
CLASSIFIER_DEVICE                             cpu
COMPLEXITY_THRESHOLD                          0.5
MODEL_NAME               worker               (required)
ENGINE_TYPE                                   ollama
INFERENCE_HOST                                http://localhost:11434
"""
from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from sqlalchemy.engine import URL


class _Settings(BaseSettings):
    # extra="ignore": one shared .env holds the keys of every class below
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class PostgresSettings(_Settings):
    """Postgres connection (POSTGRES_* env vars)."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    host: str = "postgres"
    port: int = 5432
    db: str = "model_registry"
    user: str
    password: SecretStr

    def _url(self, driver: str) -> "URL":
        # imported here: gateway and worker images ship without SQLAlchemy
        from sqlalchemy.engine import URL

        # URL.create escapes special characters in the password (@, /, :, ...)
        return URL.create(
            drivername=driver,
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.db,
        )

    @property
    def async_url(self) -> "URL":      # asyncpg, used by the application
        return self._url("postgresql+asyncpg")

    @property
    def sync_url(self) -> "URL":       # psycopg2, used by Alembic
        return self._url("postgresql+psycopg2")


class RabbitMQSettings(_Settings):
    """RabbitMQ connection + the queue the Gateway publishes to and the Router consumes."""

    model_config = SettingsConfigDict(env_prefix="RABBITMQ_")

    host: str = "rabbitmq"
    port: int = 5672
    user: str
    password: SecretStr = Field(validation_alias="RABBITMQ_PASS")
    middleware_queue: str = Field("router.requests", validation_alias="MIDDLEWARE_QUEUE")


class RouterSettings(_Settings):
    catalog_refresh_seconds: int = Field(30, validation_alias="CATALOG_REFRESH_SECONDS")


class ClassifierSettings(_Settings):
    # None -> classifier.py falls back to the baked-in checkpoint, else the HF repo
    domain_checkpoint: str | None = Field(None, validation_alias="DOMAIN_CHECKPOINT")
    complexity_checkpoint: str | None = Field(None, validation_alias="COMPLEXITY_CHECKPOINT")
    device: str = Field("cpu", validation_alias="CLASSIFIER_DEVICE")
    complexity_threshold: float = Field(0.5, validation_alias="COMPLEXITY_THRESHOLD")


class WorkerSettings(_Settings):
    model_name: str = Field(validation_alias="MODEL_NAME")
    engine_type: str = Field("ollama", validation_alias="ENGINE_TYPE")
    inference_host: str = Field("http://localhost:11434", validation_alias="INFERENCE_HOST")


@lru_cache
def get_postgres_settings() -> PostgresSettings:
    return PostgresSettings()


@lru_cache
def get_rabbitmq_settings() -> RabbitMQSettings:
    return RabbitMQSettings()


@lru_cache
def get_router_settings() -> RouterSettings:
    return RouterSettings()


@lru_cache
def get_classifier_settings() -> ClassifierSettings:
    return ClassifierSettings()


@lru_cache
def get_worker_settings() -> WorkerSettings:
    return WorkerSettings()
