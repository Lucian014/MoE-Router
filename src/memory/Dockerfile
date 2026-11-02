FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY src/memory/requirements.txt ./memory/requirements.txt
RUN pip install --no-cache-dir -r memory/requirements.txt
COPY src/database/requirements.txt ./database/requirements.txt
RUN pip install --no-cache-dir -r database/requirements.txt

# the reference tokenizer for chunk sizes (shared/memory/tokens.py), as in the router image
ENV TOKENIZER_FILE=/app/tokenizer/tokenizer.json
RUN python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('Qwen/Qwen2.5-0.5B-Instruct', 'tokenizer.json', local_dir='/app/tokenizer')"

COPY src/shared ./shared
COPY src/database ./database
COPY src/engine ./engine
COPY src/memory ./memory

# fails the build if an import is broken (needs no env) or the tokenizer / stemmer don't load
RUN python -c "import memory.service; from shared.memory.tokens import _tokenizer; assert _tokenizer(); \
from shared.memory.analyzer import analyze; assert analyze('funcția')"

CMD ["python", "-m", "memory.service"]
