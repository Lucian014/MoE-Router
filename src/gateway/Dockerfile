FROM node:22-alpine AS ui
WORKDIR /ui
COPY src/ui/package.json src/ui/package-lock.json ./
RUN npm ci
COPY src/ui ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

COPY src/gateway/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/shared ./shared
COPY src/gateway/main.py ./
COPY --from=ui /ui/dist ./static

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
