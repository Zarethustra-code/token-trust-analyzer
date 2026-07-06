# syntax=docker/dockerfile:1

# Token Trust Analyzer — production image (packaging only; no app behavior changes).
#
# Runs the FastAPI pipeline (collect -> features -> score -> report). The Isolation
# Forest fits from data/training_tokens.json during startup, so no pre-trained model
# artifact is shipped — it is rebuilt fresh inside the container on first boot.

FROM python:3.12-slim

# libgomp1 is the OpenMP runtime that scikit-learn and xgboost link against. The
# slim base image doesn't include it, and without it `import sklearn` / model
# fitting fails at startup with "libgomp.so.1: cannot open shared object file".
# It's a small runtime shared library, NOT a compiler: the app's dependencies
# install from prebuilt manylinux wheels (numpy/scipy/scikit-learn/web3/xgboost),
# so no build toolchain is needed. (If a wheel is ever missing for your platform,
# add a builder stage with build-essential and copy the site-packages over; keep
# this runtime image lean otherwise.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# --- dependency layer -------------------------------------------------------
# Copy only requirements first so that editing app code doesn't invalidate the
# (slow) pip install layer. Production deps only — requirements-dev.txt is for CI.
COPY requirements.txt requirements-slm.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Optional: local AI-content detection (AI_DETECTOR_BACKEND=local, the default).
# The deps (transformers + torch) and model weights are multi-GB, so they are
# deliberately NOT baked into the default image — without them the detector
# degrades gracefully to checked=false and everything else works. To enable:
#   docker build --build-arg INSTALL_SLM=true -t token-trust-analyzer .
# and mount a Hugging Face cache so weights download once, not per container:
#   docker run -v hf-cache:/home/appuser/.cache/huggingface ...
ARG INSTALL_SLM=false
RUN if [ "$INSTALL_SLM" = "true" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu \
        && pip install -r requirements-slm.txt; \
    fi

# --- application layer ------------------------------------------------------
# Copy the project. .dockerignore keeps out .env / secrets, the venv, tests, CI,
# docs and the cached *.joblib model (so it refits fresh here). data/ ships with
# it, so data/training_tokens.json + data/seed_tokens.json are present for the fit.
COPY . .

# Run as an unprivileged user (own the app dir so it can write the refit joblib).
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The server only starts accepting requests after the lifespan startup (which
# fits the Isolation Forest) completes, so a 200 from /health means the model is
# ready. Uses stdlib urllib — the slim base has no curl, and adding it just for a
# healthcheck isn't worth the bytes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

# app.py's __main__ reads HOST/PORT from the environment and starts uvicorn
# (reload disabled). Honors a runtime `-e PORT=...` override.
CMD ["python", "app.py"]
