def model_to_queue(model: str) -> str:
    """
    Convert a model name into the corresponding worker queue.

    Example:
        qwen2.5:7b
        ->
        worker.qwen2_5_7b.requests
    """
    normalized = model.replace(":", "_").replace(".", "_").replace("/", "_").replace("-", "_")
    return f"worker.{normalized}.requests"
