# Zero-Trust RAG demo API service.
#
# Single-stage Python 3.11 slim image. Uvicorn serves the FastAPI app
# defined in ``api.main``. The HuggingFace embedder is *not* baked into
# the image — it lazy-downloads on first query into a volume-mounted
# cache (see ``docker-compose.yml``: the ``hf_cache`` named volume),
# so rebuilds stay fast and image size stays small (~600MB).
#
# Build:
#   docker compose build api
# Run (full stack):
#   docker compose up -d
# Run (api only — assumes chromadb + ollama are already up):
#   docker compose up -d api

FROM python:3.11-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Tools that several Python deps need at install or runtime:
#  * `build-essential` for any wheel that compiles from source
#  * `curl` for the healthcheck in compose
#  * `libgomp1` for torch / sentence-transformers on slim images
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so the layer is cached across code
# changes — only the small `pip install` step re-runs when `api/*.py`
# changes.
COPY requirements.txt ./
# Generous retries + timeout because the wheel set includes heavy
# packages (torch, transformers, chromadb) and PyPI mirrors occasionally
# stall mid-download in flaky network environments.
RUN pip install --retries 5 --timeout 120 -r requirements.txt

# Copy source last. The compose service bind-mounts `./` over /app for
# dev workflows; for production builds the COPY here is the source of
# truth.
COPY . .

EXPOSE 8000

# Bind to 0.0.0.0 inside the container so the host's port mapping
# (`8000:8000` in compose) reaches uvicorn. Settings.api_host (default
# 127.0.0.1) is overridden via the API_HOST env var in compose.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
