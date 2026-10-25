#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Starting Production RAG Full-Stack Ecosystem"
echo "=========================================================="

if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found at ./venv"
    exit 1
fi

source venv/bin/activate

if [ ! -f "data/processed/chunks.json" ]; then
    echo "[INFO] Ingesting documents..."
    python -m ingest.ingest
fi

echo "[1/2] Starting FastAPI Backend on http://localhost:8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

trap "kill $BACKEND_PID" EXIT

sleep 2

echo "[2/2] Starting Next.js Frontend on http://localhost:3000..."
cd frontend
npm run dev
