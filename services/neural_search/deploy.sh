#!/bin/bash
# Deploy the Neural Search API to Google Cloud Run
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - Docker or Cloud Build enabled
#   - GOOGLE_CLOUD_PROJECT env var set
#
# Usage:
#   ./deploy.sh [--region us-central1]

set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-gen-lang-client-0470307714}"
REGION="${1:-us-central1}"
SERVICE_NAME="neural-search-api"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=== Deploying Neural Search API ==="
echo "  Project: ${PROJECT_ID}"
echo "  Region:  ${REGION}"
echo "  Service: ${SERVICE_NAME}"

# Build and push the container
echo ""
echo "Building container..."
gcloud builds submit \
  --project="${PROJECT_ID}" \
  --tag="${IMAGE}" \
  services/neural_search/

# Deploy to Cloud Run
echo ""
echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --platform=managed \
  --allow-unauthenticated \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=5 \
  --timeout=60 \
  --set-env-vars="LOG_QUERIES=true"

# Get the service URL
URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format="value(status.url)")

echo ""
echo "=== Deployment Complete ==="
echo "  Service URL: ${URL}"
echo "  Health check: ${URL}/health"
echo ""
echo "Test with:"
echo "  curl -X POST ${URL}/search \\"
echo "    -H 'Authorization: Key YOUR_API_KEY' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\": \"knowledge graph\", \"limit\": 5}'"
