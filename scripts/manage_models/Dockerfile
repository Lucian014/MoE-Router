FROM python:3.11-slim

WORKDIR /app

COPY scripts/manage_models/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

ENTRYPOINT ["python"]