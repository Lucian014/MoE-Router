
def sanitize_k8s_name(model_id: str) -> str:
    """
    Convert a catalog `model_id` (e.g. 'qwen2.5:7b' for Ollama, or
    'meta-llama/Llama-3.1-8B-Instruct' for vLLM) into a name that is safe
    to use as a Kubernetes resource name / label value.

    Kubernetes names must be lowercase RFC 1123 subdomains (alphanumerics
    and '-' only), so ':', '.', and '/' are replaced with '-'.

        'qwen2.5:7b'                         -> 'qwen2-5-7b'
        'deepseek-coder:6.7b'                -> 'deepseek-coder-6-7b'
        'meta-llama/Llama-3.1-8B-Instruct'    -> 'meta-llama-llama-3-1-8b-instruct'

    This function is deliberately engine-agnostic — it knows nothing about
    Ollama or vLLM, only about what Kubernetes will accept.
    """
    return (
        model_id.strip()
        .lower()
        .replace(":", "-")
        .replace(".", "-")
        .replace("/", "-")
    )


def model_to_queue(model_id: str) -> str:
#Deterministically map a model_id to its dedicated worker queue.

    return f"worker.{sanitize_k8s_name(model_id)}"