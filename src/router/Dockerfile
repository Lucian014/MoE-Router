# The router image ships WITHOUT the classifier weights: it is public on GHCR, while the
# checkpoints are private on Hugging Face. In the pod, an init container running this same
# image (python router/download_checkpoints.py, with HF_TOKEN from a Secret and
# HF_HUB_OFFLINE=0) downloads them into a volume that the router container mounts.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

# CPU-only torch first (the default wheel would pull ~2 GB of CUDA libraries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY src/router/requirements.txt ./router/requirements.txt
RUN pip install --no-cache-dir -r router/requirements.txt

COPY src/database/requirements.txt ./database/requirements.txt
RUN pip install --no-cache-dir -r database/requirements.txt

COPY src/shared ./shared
COPY src/database ./database
COPY src/router ./router

# fails the build if an import is broken (needs no env, no weights)
RUN python -c "import router.router"
# the router itself never downloads anything; the init container overrides this
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

CMD ["python", "-m", "router.router"]
