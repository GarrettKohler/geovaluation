FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies for geopandas, shapely, GDAL, and blob downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY src/ src/
COPY site_scoring/ site_scoring/
COPY templates/ templates/
COPY scripts/ scripts/

# Copy small processed data files (training data, summaries)
COPY data/processed/ data/processed/

# Create directories for input data and model outputs
RUN mkdir -p data/input site_scoring/outputs/experiments

# Startup script (downloads large CSV from blob storage, then starts gunicorn)
COPY startup.sh .
RUN chmod +x startup.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default: start gunicorn directly (no blob download needed if data is mounted)
# Override with startup.sh to download from Azure Blob Storage:
#   CMD ["./startup.sh"]
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "600", \
     "--workers", "2", \
     "--threads", "4", \
     "--worker-class", "gthread", \
     "app:app"]
