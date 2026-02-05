#!/usr/bin/env bash
set -e

echo "Starting Voice Detector API..."

exec uvicorn backend.app:app \
  --host 0.0.0.0 \
  --port ${PORT:-8000} \
  --workers 1
