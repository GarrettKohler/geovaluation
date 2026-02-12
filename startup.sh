#!/bin/bash
# =============================================================================
# Container Startup Script
# =============================================================================
# Downloads the large CSV from Azure Blob Storage before starting gunicorn.
# Set these environment variables in your Container App:
#   AZURE_STORAGE_ACCOUNT  - Storage account name
#   AZURE_STORAGE_KEY      - Storage account key
#   AZURE_STORAGE_CONTAINER - Blob container name (default: sitedata)
# =============================================================================

set -euo pipefail

DATA_DIR="/app/data/input"
CONTAINER="${AZURE_STORAGE_CONTAINER:-sitedata}"
MAIN_CSV="site_scores_revenue_and_diagnostics.csv"

mkdir -p "$DATA_DIR"

# Download data if not already present (supports container restart without re-download)
if [ ! -f "$DATA_DIR/$MAIN_CSV" ]; then
    echo "Downloading $MAIN_CSV from Azure Blob Storage..."

    # Use azcopy if available (faster for large files), otherwise fall back to curl with SAS
    if command -v azcopy &> /dev/null; then
        azcopy copy \
            "https://${AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}/${MAIN_CSV}" \
            "$DATA_DIR/$MAIN_CSV" \
            --log-level ERROR
    else
        # Direct download via REST API with storage key
        DATE=$(date -u +"%a, %d %b %Y %H:%M:%S GMT")
        curl -sS -o "$DATA_DIR/$MAIN_CSV" \
            -H "x-ms-date: $DATE" \
            -H "x-ms-version: 2020-10-02" \
            "https://${AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}/${MAIN_CSV}"
    fi

    echo "Download complete: $(du -h "$DATA_DIR/$MAIN_CSV" | cut -f1)"

    # Download auxiliary CSVs
    for csv in nearest_site_distances.csv site_interstate_distances.csv \
               site_kroger_distances.csv site_mcdonalds_distances.csv; do
        if [ ! -f "$DATA_DIR/$csv" ]; then
            echo "  Downloading $csv..."
            curl -sS -o "$DATA_DIR/$csv" \
                "https://${AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}/${csv}" \
                2>/dev/null || echo "  Warning: $csv not found in blob storage"
        fi
    done
else
    echo "Data files already present, skipping download."
fi

echo "Starting gunicorn..."
exec gunicorn \
    --bind 0.0.0.0:8080 \
    --timeout 600 \
    --workers 2 \
    --threads 4 \
    --worker-class gthread \
    app:app
