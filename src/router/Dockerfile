FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY src/router/requirements.txt ./router/requirements.txt
RUN pip install --no-cache-dir -r router/requirements.txt

COPY src/shared ./shared
COPY src/router ./router

CMD ["python", "-m", "router.router"]