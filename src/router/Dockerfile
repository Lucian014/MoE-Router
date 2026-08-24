FROM python:3.11-slim

WORKDIR /app

# Install dependencies (reusing worker requirements as they are likely similar)
COPY src/worker/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared code and service code
COPY src/shared ./shared
COPY src/router/router.py ./

CMD ["python", "router.py"]
