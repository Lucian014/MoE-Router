"""The inference adapters (src/engine/), against a mocked HTTP server: what they send, what they parse."""
import asyncio
import json

import httpx
import pytest

from engine import ChatResult, OllamaEngine, VLLMEngine, get_engine

MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "salut"}]


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=10))


def server(response: dict, status: int = 200):
    """A MockTransport that answers every request with `response` and records the requests."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=response)

    return httpx.MockTransport(handler), seen


# ------------------------------------------------------------------ Ollama
def test_ollama_chat_sends_messages_and_limits():
    transport, seen = server({"message": {"role": "assistant", "content": "Bună!"},
                              "prompt_eval_count": 21, "eval_count": 3, "done_reason": "stop"})
    result = run(OllamaEngine("http://ollama:11434", transport).chat("gemma3:1b", MESSAGES, 512, 4096))

    assert result == ChatResult(text="Bună!", prompt_tokens=21, completion_tokens=3, truncated=False)
    (req,) = seen
    assert req.url == "http://ollama:11434/api/chat"
    body = json.loads(req.content)
    assert body == {"model": "gemma3:1b", "messages": MESSAGES, "stream": False,
                    "options": {"num_ctx": 4096, "num_predict": 512}}


def test_ollama_reports_truncation():
    transport, _ = server({"message": {"content": "```python\ndef f("}, "done_reason": "length"})
    result = run(OllamaEngine("http://o", transport).chat("m", MESSAGES, 8, 4096))
    assert result.truncated and result.prompt_tokens is None


def test_ollama_http_error_raises():
    transport, _ = server({"error": "model not found"}, status=404)
    with pytest.raises(httpx.HTTPStatusError):
        run(OllamaEngine("http://o", transport).chat("m", MESSAGES, 8, 4096))


def test_ollama_health_checks_the_model_is_pulled():
    transport, _ = server({"models": [{"name": "gemma3:1b"}, {"name": "llama3.2:latest"}]})
    engine = OllamaEngine("http://o", transport)
    run(engine.health("gemma3:1b"))
    run(engine.health("llama3.2"))                       # pulled without a tag -> ':latest'
    with pytest.raises(RuntimeError, match="not pulled"):
        run(engine.health("qwen2.5:3b"))


# ------------------------------------------------------------------ vLLM
def test_vllm_chat_uses_the_openai_chat_endpoint():
    transport, seen = server({"choices": [{"message": {"content": "Bună!"}, "finish_reason": "length"}],
                              "usage": {"prompt_tokens": 30, "completion_tokens": 512}})
    result = run(VLLMEngine("http://vllm:8000", transport).chat("qwen", MESSAGES, 512, 4096))

    assert result == ChatResult(text="Bună!", prompt_tokens=30, completion_tokens=512, truncated=True)
    (req,) = seen
    assert req.url == "http://vllm:8000/v1/chat/completions"
    assert json.loads(req.content) == {"model": "qwen", "messages": MESSAGES, "max_tokens": 512}


def test_vllm_health_checks_the_served_model():
    transport, _ = server({"data": [{"id": "qwen"}]})
    run(VLLMEngine("http://v", transport).health("qwen"))
    with pytest.raises(RuntimeError, match="not served"):
        run(VLLMEngine("http://v", transport).health("other"))


def test_get_engine_rejects_unknown_types():
    assert isinstance(get_engine("ollama", "http://o"), OllamaEngine)
    with pytest.raises(ValueError, match="Unknown ENGINE_TYPE"):
        get_engine("tgi", "http://t")
