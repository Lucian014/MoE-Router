from abc import ABC, abstractmethod


class InferenceEngine(ABC):
    """
    Common interface between the Worker and whatever actually runs the
    model. The Worker only ever talks to this interface — it never knows
    or cares whether requests end up at Ollama, vLLM, or anything added
    later. Adding a new backend means writing one new adapter here, not
    touching worker.py or the message-handling logic.
    """

    @abstractmethod
    async def generate(self, model: str, prompt: str) -> str:
        """Send `prompt` to `model` and return the generated text."""
        raise NotImplementedError

    @abstractmethod
    async def health(self, model: str) -> None:
        """Return if the engine is up and can serve `model`; raise (with the reason) otherwise.
        The worker only consumes / reports ready while this passes."""
        raise NotImplementedError
