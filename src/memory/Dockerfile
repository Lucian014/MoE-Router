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
# only tokens.py is copied here (REFERENCE_TOKENIZER), so the layer is rebuilt only when it changes
COPY src/shared/memory/tokens.py /tmp/tokens.py
RUN python /tmp/tokens.py /app/tokenizer && rm /tmp/tokens.py

COPY src/shared ./shared
COPY src/database ./database
COPY src/memory ./memory

# fails the build if an import is broken (needs no env) or the tokenizer / stemmer don't load
RUN python -c "import memory.service; from shared.memory.tokens import _tokenizer; assert _tokenizer(); \
from shared.memory.analyzer import analyze; assert analyze('funcția')"

CMD ["python", "-m", "memory.service"]
