"""The central settings module (src/shared/config.py)."""
import pytest
from pydantic import ValidationError

from shared import config
from shared.config import (
    ClassifierSettings,
    GatewaySettings,
    LifecycleSettings,
    PostgresSettings,
    RabbitMQSettings,
    RouterSettings,
    WorkerSettings,
)

ALL_VARS = [
    "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD",
    "RABBITMQ_HOST", "RABBITMQ_PORT", "RABBITMQ_USER", "RABBITMQ_PASS", "RABBITMQ_PASSWORD",
    "MIDDLEWARE_QUEUE", "RPC_TIMEOUT_SECONDS", "CATALOG_REFRESH_SECONDS",
    "DOMAIN_CHECKPOINT", "COMPLEXITY_CHECKPOINT", "CLASSIFIER_DEVICE", "COMPLEXITY_THRESHOLD",
    "MODEL_NAME", "ENGINE_TYPE", "INFERENCE_HOST", "WORKER_PREFETCH",
    "RABBITMQ_CONNECT_RETRIES", "READY_FILE", "SHUTDOWN_GRACE_SECONDS",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """No inherited env vars, no .env from the repo, no cached settings between tests."""
    for var in ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    for getter in (config.get_postgres_settings, config.get_rabbitmq_settings, config.get_gateway_settings,
                   config.get_router_settings, config.get_classifier_settings, config.get_worker_settings,
                   config.get_lifecycle_settings):
        getter.cache_clear()
    yield


# ------------------------------------------------------------------ Postgres
def test_postgres_defaults(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    s = PostgresSettings()
    assert (s.host, s.port, s.db) == ("postgres", 5432, "model_registry")


def test_postgres_user_and_password_are_required():
    with pytest.raises(ValidationError) as exc:
        PostgresSettings()
    missing = {err["loc"][0] for err in exc.value.errors()}
    assert missing == {"user", "password"}


def test_postgres_url_escapes_special_characters(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p%@/:x")
    s = PostgresSettings()
    assert s.sync_url.render_as_string(hide_password=False) == \
        "postgresql+psycopg2://u:p%25%40%2F%3Ax@postgres:5432/model_registry"
    assert s.async_url.drivername == "postgresql+asyncpg"
    assert s.async_url.password == "p%@/:x"


def test_postgres_password_is_hidden_in_repr(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "hunter2")
    assert "hunter2" not in repr(PostgresSettings())


# ------------------------------------------------------------------ RabbitMQ
def test_rabbitmq_password_comes_from_rabbitmq_pass(monkeypatch):
    # the manifests set RABBITMQ_PASS, not RABBITMQ_PASSWORD
    monkeypatch.setenv("RABBITMQ_USER", "u")
    monkeypatch.setenv("RABBITMQ_PASS", "secret")
    s = RabbitMQSettings()
    assert s.password.get_secret_value() == "secret"
    assert (s.host, s.port, s.middleware_queue) == ("rabbitmq", 5672, "router.requests")


def test_rabbitmq_password_ignores_rabbitmq_password(monkeypatch):
    monkeypatch.setenv("RABBITMQ_USER", "u")
    monkeypatch.setenv("RABBITMQ_PASSWORD", "wrong-name")
    with pytest.raises(ValidationError):
        RabbitMQSettings()


def test_rabbitmq_middleware_queue_override(monkeypatch):
    monkeypatch.setenv("RABBITMQ_USER", "u")
    monkeypatch.setenv("RABBITMQ_PASS", "p")
    monkeypatch.setenv("MIDDLEWARE_QUEUE", "other.queue")
    assert RabbitMQSettings().middleware_queue == "other.queue"


def test_rabbitmq_connect_retries(monkeypatch):
    monkeypatch.setenv("RABBITMQ_USER", "u")
    monkeypatch.setenv("RABBITMQ_PASS", "p")
    assert RabbitMQSettings().connect_retries == 0          # forever
    monkeypatch.setenv("RABBITMQ_CONNECT_RETRIES", "3")
    assert RabbitMQSettings().connect_retries == 3
    monkeypatch.setenv("RABBITMQ_CONNECT_RETRIES", "-1")
    with pytest.raises(ValidationError):
        RabbitMQSettings()


# ------------------------------------------------------------------ services
def test_lifecycle_settings(monkeypatch):
    s = LifecycleSettings()
    assert (s.ready_file, s.shutdown_grace_seconds) == ("/tmp/ready", 25)
    monkeypatch.setenv("SHUTDOWN_GRACE_SECONDS", "80")
    assert LifecycleSettings().shutdown_grace_seconds == 80



def test_gateway_timeout(monkeypatch):
    assert GatewaySettings().rpc_timeout_seconds == 300
    monkeypatch.setenv("RPC_TIMEOUT_SECONDS", "2.5")
    assert GatewaySettings().rpc_timeout_seconds == 2.5
    monkeypatch.setenv("RPC_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValidationError):
        GatewaySettings()


def test_router_and_classifier_defaults():
    assert RouterSettings().catalog_refresh_seconds == 30
    c = ClassifierSettings()
    assert (c.domain_checkpoint, c.complexity_checkpoint, c.device, c.complexity_threshold) == \
        (None, None, "cpu", 0.5)


def test_classifier_threshold_is_parsed_as_float(monkeypatch):
    monkeypatch.setenv("COMPLEXITY_THRESHOLD", "0.3")
    assert ClassifierSettings().complexity_threshold == 0.3


def test_worker_requires_model_name():
    with pytest.raises(ValidationError, match="MODEL_NAME"):
        WorkerSettings()


def test_worker_settings(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "gemma3:1b")
    s = WorkerSettings()
    assert (s.model_name, s.engine_type, s.inference_host, s.prefetch) == \
        ("gemma3:1b", "ollama", "http://localhost:11434", 1)
    monkeypatch.setenv("WORKER_PREFETCH", "0")
    with pytest.raises(ValidationError):
        WorkerSettings()


# ------------------------------------------------------------------ sources and caching
def test_dotenv_file_is_read_and_unknown_keys_are_ignored(tmp_path):
    (tmp_path / ".env").write_text("RABBITMQ_USER=fromfile\nRABBITMQ_PASS=x\nSOMETHING_ELSE=1\n")
    assert RabbitMQSettings().user == "fromfile"


def test_real_env_wins_over_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("RABBITMQ_USER=fromfile\nRABBITMQ_PASS=x\n")
    monkeypatch.setenv("RABBITMQ_USER", "fromenv")
    assert RabbitMQSettings().user == "fromenv"


def test_getters_are_cached(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "a")
    first = config.get_worker_settings()
    monkeypatch.setenv("MODEL_NAME", "b")
    assert config.get_worker_settings() is first


def test_services_import_without_any_env():
    # the router image imports router.router at build time, with no env at all
    import database  # noqa: F401
    import router.router  # noqa: F401
