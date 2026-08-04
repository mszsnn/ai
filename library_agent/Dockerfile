FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    VECTOR_DB_PATH=/data/chroma \
    CHECKPOINT_DB_PATH=/data/checkpoints.sqlite \
    LIBRARY_DB_PATH=/data/library.sqlite3 \
    UPLOAD_DIR=/data/uploads

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY library_agent/requirements.txt /app/library_agent/requirements.txt
RUN pip install --no-cache-dir -r /app/library_agent/requirements.txt

COPY library_agent/api /app/library_agent/api
COPY library_agent/agent /app/library_agent/agent
COPY library_agent/infrastructure /app/library_agent/infrastructure
COPY library_agent/rag_engine /app/library_agent/rag_engine
COPY library_agent/deploy/docker-entrypoint.sh /usr/local/bin/library-agent-entrypoint

RUN mkdir -p /data \
    && chmod +x /usr/local/bin/library-agent-entrypoint \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /data /usr/local/bin/library-agent-entrypoint

VOLUME ["/data"]
EXPOSE 8000

USER appuser
ENTRYPOINT ["/usr/local/bin/library-agent-entrypoint"]
CMD ["uvicorn", "library_agent.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
