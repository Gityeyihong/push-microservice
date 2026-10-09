FROM python:3.11-slim AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin push

COPY --from=build /usr/local /usr/local
COPY . .

RUN chmod +x /app/scripts/docker-entrypoint.sh \
    && python scripts/download_embedding_model.py --method hub \
    && python -m data.generate_data \
    && mkdir -p /app/data /app/reports /app/config \
    && chown -R push:push /app

EXPOSE 8000

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
