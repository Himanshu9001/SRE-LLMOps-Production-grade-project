#!/bin/bash
# P1 Data Pipeline Runner
# Runs: scrape → format → validate → upload to S3
# Usage: bash scripts/run_pipeline.sh [--skip-scrape] [--skip-upload]

set -euo pipefail

SKIP_SCRAPE=false
SKIP_UPLOAD=false

for arg in "$@"; do
  case $arg in
    --skip-scrape) SKIP_SCRAPE=true ;;
    --skip-upload) SKIP_UPLOAD=true ;;
  esac
done

echo "========================================="
echo " SRE LLMOps — P1 Data Pipeline"
echo "========================================="

if [ "$SKIP_SCRAPE" = false ]; then
  echo "[1/4] Running scrapers..."
  python -m src.pipeline.run_pipeline --stage scrape
else
  echo "[1/4] Skipping scrape (--skip-scrape)"
fi

echo "[2/4] Formatting raw data to Alpaca JSONL..."
python -m src.pipeline.run_pipeline --stage format

echo "[3/4] Validating and deduplicating dataset..."
python -m src.pipeline.run_pipeline --stage validate

if [ "$SKIP_UPLOAD" = false ]; then
  echo "[4/4] Uploading to S3 + DVC tracking..."
  python -m src.pipeline.run_pipeline --stage upload
  dvc add data/validated/
  dvc push
else
  echo "[4/4] Skipping upload (--skip-upload)"
fi

echo ""
echo "========================================="
echo " P1 Complete. Dataset ready for P4."
echo "========================================="
