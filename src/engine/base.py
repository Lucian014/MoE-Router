from abc import ABC, abstractmethod
from dataclasses import dataclass

# a chat message: {"role": "system" | "user" | "assistant", "content": str}
Message = dict[str, str]


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int | None = None        # as counted by the engine, when it reports them
    completion_tokens: int | None = None
    truncated: bool = False                 # stopped at max_tokens, not at the end of the answer


class InferenceEngine(ABC):
    """
    Common interface between the Worker and whatever actually runs the
    model. The Worker only ever talks to this interface — it never knows
    or cares whether requests end up at Ollama, vLLM, or anything added
    later. Adding a new backend means writing one new adapter here, not
    touching worker.py or the message-handling logic.

    Engines take a list of chat messages, not a raw prompt: each server
    applies the model's own chat template, and the conversation context
    (system prompt, earlier turns, retrieved fragments) travels as messages.
    """

    @abstractmethod
    async def chat(self, model: str, messages: list[Message], max_tokens: int, num_ctx: int) -> ChatResult:
        """Send `messages` to `model` and return the answer.

        max_tokens: upper bound for the answer.
        num_ctx:    context window for this request, input + output. Engines that fix it at
                    server start (vLLM --max-model-len) ignore it."""
        raise NotImplementedError

    @abstractmethod
    async def health(self, model: str) -> None:
        """Return if the engine is up and can serve `model`; raise (with the reason) otherwise.
        The worker only consumes / reports ready while this passes."""
        raise NotImplementedError
