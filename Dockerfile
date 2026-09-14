FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FOODVISION_CACHE_DIR=/opt/foodvision/cache

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch wheels: roughly 800MB smaller than the default CUDA build,
# which matters because nothing in the served path needs a GPU.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY app ./app
COPY model ./model

# The checkpoint is fetched from the GitHub release on first request and cached
# here. Bake it into the image instead by mounting or copying a local file and
# setting FOODVISION_WEIGHTS_PATH.
RUN mkdir -p /opt/foodvision/cache \
    && useradd --create-home --uid 10001 foodvision \
    && chown -R foodvision /opt/foodvision /srv
USER foodvision

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
