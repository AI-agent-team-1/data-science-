#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/LLM_bot}"
cd "$APP_DIR"

echo "[deploy] Pull latest code"
git pull --ff-only

if ! command -v docker >/dev/null 2>&1; then
  echo "[deploy] Docker not found. Install it first." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "[deploy] docker compose plugin not found. Install docker-compose-plugin." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "[deploy] .env not found in $APP_DIR" >&2
  echo "[deploy] Create it on the VPS (do NOT commit it). Example in env.example" >&2
  exit 1
fi

mkdir -p docs rag_faiss_index

echo "[deploy] Build & start"
docker compose up -d --build

echo "[deploy] Done. Status:"
docker ps --filter "name=llm-bot"
