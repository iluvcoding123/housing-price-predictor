# ---------- Base image ----------
FROM python:3.12-slim AS base

# Prevent Python from writing pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (add git if you install from private repos)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash appuser
WORKDIR /app

# ---------- Dependency layer (better caching) ----------
# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# ---------- Application code & artifacts ----------
# Copy src code, models and minimal data (feature_names)
COPY src ./src
COPY models ./models
COPY data/feature_names.pkl ./data/feature_names.pkl

# Make sure paths exist
RUN mkdir -p data models

# Switch to non-root user for security
USER appuser

# Expose FastAPI port
EXPOSE 8000

# ---------- Run ----------
# Use uvicorn to serve FastAPI app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
