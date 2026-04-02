FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy full project
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.1" \
    "openai>=1.0.0" \
    "python-dotenv>=1.0.0" \
    "uvicorn[standard]>=0.29.0"

# Set PYTHONPATH so server/ imports resolve correctly
ENV PYTHONPATH="/app:$PYTHONPATH"

# HF Spaces uses port 7860; local dev defaults to 8000
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "cd /app && uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
