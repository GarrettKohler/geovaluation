#!/bin/bash
# =============================================================================
# Azure Container Apps Deployment Script
# =============================================================================
# Deploys the geospatial site scoring app to Azure Container Apps.
#
# Prerequisites:
#   - Azure CLI installed: brew install azure-cli
#   - Logged in: az login
#   - Docker running locally
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
RESOURCE_GROUP="geospatial-rg"
LOCATION="eastus"
ACR_NAME="geospatialacr"          # Must be globally unique, lowercase
APP_NAME="geospatial-scoring"
STORAGE_ACCOUNT="geospatialstorage"  # Must be globally unique, lowercase
CONTAINER_NAME="sitedata"            # Blob container for the large CSV
IMAGE_TAG="latest"
# ─────────────────────────────────────────────────────────────────────────────

echo "=== Step 1: Create Resource Group ==="
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output table

echo ""
echo "=== Step 2: Create Azure Container Registry ==="
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic \
    --output table

az acr login --name "$ACR_NAME"

echo ""
echo "=== Step 3: Create Storage Account & Upload Large CSV ==="
az storage account create \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --sku Standard_LRS \
    --output table

STORAGE_KEY=$(az storage account keys list \
    --resource-group "$RESOURCE_GROUP" \
    --account-name "$STORAGE_ACCOUNT" \
    --query "[0].value" -o tsv)

az storage container create \
    --name "$CONTAINER_NAME" \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --output table

echo "Uploading large CSV to blob storage (this may take a few minutes)..."
az storage blob upload \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --container-name "$CONTAINER_NAME" \
    --name "site_scores_revenue_and_diagnostics.csv" \
    --file "data/input/site_scores_revenue_and_diagnostics.csv" \
    --overwrite \
    --output table

# Upload auxiliary input CSVs needed by ETL
for csv_file in data/input/*.csv; do
    filename=$(basename "$csv_file")
    if [ "$filename" != "site_scores_revenue_and_diagnostics.csv" ]; then
        echo "  Uploading $filename..."
        az storage blob upload \
            --account-name "$STORAGE_ACCOUNT" \
            --account-key "$STORAGE_KEY" \
            --container-name "$CONTAINER_NAME" \
            --name "$filename" \
            --file "$csv_file" \
            --overwrite \
            --output none
    fi
done

echo ""
echo "=== Step 4: Build & Push Docker Image ==="
FULL_IMAGE="${ACR_NAME}.azurecr.io/${APP_NAME}:${IMAGE_TAG}"

docker build --platform linux/amd64 -t "$FULL_IMAGE" .
docker push "$FULL_IMAGE"

echo ""
echo "=== Step 5: Create Container Apps Environment ==="
az containerapp env create \
    --name "${APP_NAME}-env" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output table

# Create Azure Files share for the input data mount
SHARE_NAME="inputdata"
az storage share create \
    --name "$SHARE_NAME" \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --output table

# Add storage to Container Apps environment
az containerapp env storage set \
    --name "${APP_NAME}-env" \
    --resource-group "$RESOURCE_GROUP" \
    --storage-name "inputdata" \
    --azure-file-account-name "$STORAGE_ACCOUNT" \
    --azure-file-account-key "$STORAGE_KEY" \
    --azure-file-share-name "$SHARE_NAME" \
    --access-mode ReadOnly \
    --output table

echo ""
echo "=== Step 6: Deploy Container App ==="
az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "${APP_NAME}-env" \
    --image "$FULL_IMAGE" \
    --registry-server "${ACR_NAME}.azurecr.io" \
    --target-port 8080 \
    --ingress external \
    --cpu 2 \
    --memory 4Gi \
    --min-replicas 1 \
    --max-replicas 3 \
    --env-vars "FLASK_ENV=production" \
    --output table

echo ""
echo "=== Step 7: Get App URL ==="
APP_URL=$(az containerapp show \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "properties.configuration.ingress.fqdn" -o tsv)

echo ""
echo "================================================"
echo "  Deployment complete!"
echo "  App URL: https://${APP_URL}"
echo "================================================"
echo ""
echo "IMPORTANT: The app needs the large CSV to be available at"
echo "data/input/site_scores_revenue_and_diagnostics.csv inside the container."
echo ""
echo "Option A (simple): Bake it into the Docker image by removing the"
echo "  data/input/ line from .dockerignore, then rebuild."
echo ""
echo "Option B (recommended): Download from blob storage on container start."
echo "  See the startup-with-data.sh script."
echo ""
echo "To update the app after code changes:"
echo "  docker build --platform linux/amd64 -t $FULL_IMAGE ."
echo "  docker push $FULL_IMAGE"
echo "  az containerapp update --name $APP_NAME --resource-group $RESOURCE_GROUP --image $FULL_IMAGE"
