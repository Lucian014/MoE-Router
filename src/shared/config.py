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
RABBITMQ_CONNECT_RETRIES                      0 (= retry forever, with backoff up to 30 s)
MIDDLEWARE_QUEUE         gateway, router      router.requests
RPC_TIMEOUT_SECONDS      gateway              300 (> the worker's engine timeout of 240 s)
CATALOG_REFRESH_SECONDS  router               30
DOMAIN_CHECKPOINT        router (classifier)  router/checkpoints/laya-domain-v4, else HF repo
COMPLEXITY_CHECKPOINT                         router/checkpoints/laya-complexity-v2, else HF repo
CLASSIFIER_DEVICE                             cpu
COMPLEXITY_THRESHOLD                          0.5
HF_TOKEN                 router init          (required: the classifier repos are private)
CHECKPOINT_DIR           (download_           /checkpoints
DOMAIN_REPO               checkpoints.py)     Lucian04/laya-domain-v4
DOMAIN_REVISION                               main (pin a commit hash for reproducible pods)
COMPLEXITY_REPO                               Lucian04/laya-complexity-v2
COMPLEXITY_REVISION                           main
MODEL_NAME               worker               (required)
ENGINE_TYPE                                   ollama
INFERENCE_HOST                                http://localhost:11434
WORKER_PREFETCH                               1 (requests handled concurrently per pod)
READY_FILE               router, worker       /tmp/ready (exists while the pod is ready)
SHUTDOWN_GRACE_SECONDS                        25 (on SIGTERM: wait this long for in-flight work)
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
    # 0 = retry forever: in Kubernetes the broker may start after the services
    connect_retries: int = Field(0, ge=0, validation_alias="RABBITMQ_CONNECT_RETRIES")


class LifecycleSettings(_Settings):
    """Readiness and shutdown of the router and worker (they have no HTTP server)."""

    # the readinessProbe is `test -f <ready_file>`
    ready_file: str = Field("/tmp/ready", validation_alias="READY_FILE")
    # keep below the pod's terminationGracePeriodSeconds
    shutdown_grace_seconds: float = Field(25, ge=0, validation_alias="SHUTDOWN_GRACE_SECONDS")


class GatewaySettings(_Settings):
    # how long a request waits for its reply; also its TTL in the queues
    rpc_timeout_seconds: float = Field(300, gt=0, validation_alias="RPC_TIMEOUT_SECONDS")


class RouterSettings(_Settings):
    catalog_refresh_seconds: int = Field(30, validation_alias="CATALOG_REFRESH_SECONDS")


class ClassifierSettings(_Settings):
    # None -> classifier.py falls back to src/router/checkpoints/<name> (local runs), else the
    # HF repo. In Kubernetes these point at the volume the init container downloaded into.
    domain_checkpoint: str | None = Field(None, validation_alias="DOMAIN_CHECKPOINT")
    complexity_checkpoint: str | None = Field(None, validation_alias="COMPLEXITY_CHECKPOINT")
    device: str = Field("cpu", validation_alias="CLASSIFIER_DEVICE")
    complexity_threshold: float = Field(0.5, validation_alias="COMPLEXITY_THRESHOLD")


class CheckpointSettings(_Settings):
    """Where router/download_checkpoints.py gets the classifiers from and puts them.
    The repos are private, so the router pod's init container needs an HF token."""

    hf_token: SecretStr = Field(validation_alias="HF_TOKEN")
    checkpoint_dir: str = Field("/checkpoints", validation_alias="CHECKPOINT_DIR")
    domain_repo: str = Field("Lucian04/laya-domain-v4", validation_alias="DOMAIN_REPO")
    domain_revision: str = Field("main", validation_alias="DOMAIN_REVISION")
    complexity_repo: str = Field("Lucian04/laya-complexity-v2", validation_alias="COMPLEXITY_REPO")
    complexity_revision: str = Field("main", validation_alias="COMPLEXITY_REVISION")


class WorkerSettings(_Settings):
    model_name: str = Field(validation_alias="MODEL_NAME")
    engine_type: str = Field("ollama", validation_alias="ENGINE_TYPE")
    inference_host: str = Field("http://localhost:11434", validation_alias="INFERENCE_HOST")
    # >1 lets an engine that batches (vLLM) work on several requests at once
    prefetch: int = Field(1, ge=1, validation_alias="WORKER_PREFETCH")


@lru_cache
def get_checkpoint_settings() -> CheckpointSettings:
    return CheckpointSettings()


@lru_cache
def get_postgres_settings() -> PostgresSettings:
    return PostgresSettings()


@lru_cache
def get_rabbitmq_settings() -> RabbitMQSettings:
    return RabbitMQSettings()


@lru_cache
def get_lifecycle_settings() -> LifecycleSettings:
    return LifecycleSettings()


@lru_cache
def get_gateway_settings() -> GatewaySettings:
    return GatewaySettings()


@lru_cache
def get_router_settings() -> RouterSettings:
    return RouterSettings()


@lru_cache
def get_classifier_settings() -> ClassifierSettings:
    return ClassifierSettings()


@lru_cache
def get_worker_settings() -> WorkerSettings:
    return WorkerSettings()
