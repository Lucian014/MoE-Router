FROM python:3.11-slim

WORKDIR /app

# Install dependencies (the database ones: the worker stores every turn)
COPY src/worker/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/database/requirements.txt ./database-requirements.txt
RUN pip install --no-cache-dir -r database-requirements.txt

# Copy shared code and service code
COPY src/shared ./shared
COPY src/database ./database
COPY src/worker/worker.py ./
COPY src/engine ./engine

# fails the build if an import is broken (needs no env)
RUN python -c "import worker"
CMD ["python", "worker.py"]
