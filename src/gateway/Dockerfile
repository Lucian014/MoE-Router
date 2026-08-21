FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY src/gateway/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared code and service code
COPY src/shared ./shared
COPY src/gateway/main.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
