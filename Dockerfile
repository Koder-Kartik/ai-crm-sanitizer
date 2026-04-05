# server/Dockerfile
# CRM Sanitizer — Container Definition
#
# This file packages the environment server into a Docker container.
# Hugging Face Spaces runs this container when your Space is deployed.
#
# Build:  docker build -t crm-sanitizer -f server/Dockerfile .
# Run:    docker run -p 7860:7860 crm-sanitizer
#
# IMPORTANT:
#   The build context is the PROJECT ROOT (crm-sanitizer/)
#   not the server/ folder. This is because we need both
#   server/ files AND models.py from the root.
#   Always build with: docker build -f server/Dockerfile .
#   from the project root directory.

# ─────────────────────────────────────────────
# BASE IMAGE
# Python 3.11 slim — small, fast, stable
# ─────────────────────────────────────────────

FROM python:3.11-slim

# ─────────────────────────────────────────────
# SYSTEM SETUP
# ─────────────────────────────────────────────

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python from buffering stdout/stderr
# Critical for seeing logs in real time on HF Spaces
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# ─────────────────────────────────────────────
# INSTALL DEPENDENCIES
# Copy requirements first — Docker caches this layer
# Only rebuilds if requirements.txt changes
# ─────────────────────────────────────────────

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# COPY APPLICATION CODE
# Copy in order: root files first, then server/
# ─────────────────────────────────────────────

# Copy models.py from project root
# server/environment.py imports this
COPY models.py .

# Copy entire server/ directory
COPY server/ ./server/

# ─────────────────────────────────────────────
# ENVIRONMENT VARIABLES
# These can be overridden at runtime
# ─────────────────────────────────────────────

# Port Hugging Face Spaces expects
ENV PORT=7860

# Host binding
ENV HOST=0.0.0.0

# Uvicorn workers
# Keep at 1 for HF free tier (2 vCPU, 8GB RAM)
ENV WORKERS=1

# ─────────────────────────────────────────────
# EXPOSE PORT
# Tells Docker which port the container listens on
# ─────────────────────────────────────────────

EXPOSE 7860

# ─────────────────────────────────────────────
# HEALTHCHECK
# Docker checks this to know if container is healthy
# Tries every 30 seconds, gives 60 seconds to start
# ─────────────────────────────────────────────

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; \
     urllib.request.urlopen('http://localhost:7860/health')" \
    || exit 1

# ─────────────────────────────────────────────
# STARTUP COMMAND
# Runs the FastAPI server with uvicorn
# ─────────────────────────────────────────────

CMD ["sh", "-c", \
     "cd /app/server && \
      python -m uvicorn app:app \
      --host 0.0.0.0 \
      --port 7860 \
      --workers 1 \
      --log-level info"]