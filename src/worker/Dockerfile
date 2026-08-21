FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY src/worker/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared code and service code
COPY src/shared ./shared
COPY src/worker/worker.py ./

CMD ["python", "worker.py"]
