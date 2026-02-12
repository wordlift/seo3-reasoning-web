#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# GCP Setup Script for Structured LD Agentic RAG
# Enables required APIs and creates Vertex AI Vector Search 2.0 collections.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-gen-lang-client-0470307714}"
REGION="${GCP_REGION:-us-central1}"

echo "=== Configuring GCP Project: ${PROJECT_ID} ==="
gcloud config set project "${PROJECT_ID}"

# ---------------------------------------------------------------------------
# Enable APIs
# ---------------------------------------------------------------------------
echo "=== Enabling APIs ==="
gcloud services enable \
    aiplatform.googleapis.com \
    compute.googleapis.com \
    storage.googleapis.com \
    cloudbuild.googleapis.com

echo "=== APIs enabled ==="

# ---------------------------------------------------------------------------
# Create GCS bucket for experiment data (if needed)
# ---------------------------------------------------------------------------
BUCKET_NAME="gs://${PROJECT_ID}-sld-agentic-rag"
if ! gsutil ls "${BUCKET_NAME}" &>/dev/null; then
    echo "=== Creating GCS bucket: ${BUCKET_NAME} ==="
    gsutil mb -l "${REGION}" "${BUCKET_NAME}"
else
    echo "=== GCS bucket already exists: ${BUCKET_NAME} ==="
fi

echo ""
echo "=== GCP setup complete ==="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Bucket:   ${BUCKET_NAME}"
echo ""
echo "Next steps:"
echo "  1. Run 'python -m src.dataset.collector' to fetch entity data"
echo "  2. Run 'python -m src.indexing.vectorsearch --setup' to create Vector Search collections"
