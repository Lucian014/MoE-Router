
# ---------- stage 1: download the checkpoints --------------------------------------------------
FROM python:3.11-slim AS checkpoints

ARG DOMAIN_REPO=Lucian04/laya-domain-v4
ARG COMPLEXITY_REPO=Lucian04/laya-complexity-v2
# pin a commit hash for reproducible builds (default: latest commit on main)
ARG DOMAIN_REVISION=main
ARG COMPLEXITY_REVISION=main

RUN pip install --no-cache-dir "huggingface_hub>=0.23"

COPY src/router/download_checkpoints.py /download_checkpoints.py
RUN --mount=type=secret,id=hf_token \
    DOMAIN_REPO=${DOMAIN_REPO} DOMAIN_REVISION=${DOMAIN_REVISION} \
    COMPLEXITY_REPO=${COMPLEXITY_REPO} COMPLEXITY_REVISION=${COMPLEXITY_REVISION} \
    python /download_checkpoints.py

# ---------- stage 2: the router -------------------------------------------------------------------
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
COPY --from=checkpoints /checkpoints ./router/checkpoints

RUN python -c "from router.classifier import load_classifiers; load_classifiers()"
RUN python -c "import router.router"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

CMD ["python", "-m", "router.router"]