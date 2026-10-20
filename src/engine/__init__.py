from .base import InferenceEngine
from .ollama_engine import OllamaEngine
from .vllm_engine import VLLMEngine

# Keep this in sync with the engines the Helm chart accepts
# (charts/moe-router/templates/catalog-validate.yaml: ollama, vllm).
_ENGINES = {
    "ollama": OllamaEngine,
    "vllm": VLLMEngine,
}


def get_engine(engine_type: str, host: str) -> InferenceEngine:
    """Instantiate the adapter for a given ENGINE_TYPE env var value."""
    try:
        engine_cls = _ENGINES[engine_type]
    except KeyError:
        raise ValueError(
            f"Unknown ENGINE_TYPE '{engine_type}'. Supported engines: {sorted(_ENGINES)}"
        )
    return engine_cls(host)
