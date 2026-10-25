#!/usr/bin/env python3
"""
Automated Environment & Credentials Configurator
Safely updates .env and frontend/.env.local with provided production keys.
"""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_ENV = ROOT_DIR / ".env"
FRONTEND_ENV = ROOT_DIR / "frontend" / ".env.local"


def update_env_file(file_path: Path, updates: dict):
    if not file_path.exists():
        existing_lines = []
    else:
        existing_lines = file_path.read_text(encoding="utf-8").splitlines()

    updated_keys = set()
    new_lines = []

    for line in existing_lines:
        line_stripped = line.strip()
        if line_stripped.startswith("#") or "=" not in line_stripped:
            new_lines.append(line)
            continue

        key = line_stripped.split("=", 1)[0].strip()
        if key in updates and updates[key] is not None:
            new_lines.append(f"{key}={updates[key]}")
            updated_keys.add(key)
        else:
            new_lines.append(line)

    for key, val in updates.items():
        if key not in updated_keys and val is not None:
            new_lines.append(f"{key}={val}")

    file_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"[OK] Updated {file_path.name} ({len(updates)} key(s) modified)")


def main():
    parser = argparse.ArgumentParser(description="Configure production credentials for RAG system.")
    parser.add_argument("--openai-key", help="OpenAI API Key (sk-...)")
    parser.add_argument("--gemini-key", help="Google Gemini API Key (AIzaSy...)")
    parser.add_argument("--qdrant-url", help="Qdrant Cloud Cluster URL (https://xxx.qdrant.io:6333)")
    parser.add_argument("--qdrant-key", help="Qdrant Cloud API Key")
    parser.add_argument("--redis-url", help="Upstash / Redis URL (rediss://...)")
    parser.add_argument("--langfuse-public", help="Langfuse Public Key (pk-lf-...)")
    parser.add_argument("--langfuse-secret", help="Langfuse Secret Key (sk-lf-...)")
    parser.add_argument("--api-key", help="Application API Secret Key for auth")

    args = parser.parse_args()

    backend_updates = {}
    if args.openai_key:
        backend_updates["OPENAI_API_KEY"] = args.openai_key
        backend_updates["LLM_PROVIDER"] = "openai"
    if args.gemini_key:
        backend_updates["GEMINI_API_KEY"] = args.gemini_key
        backend_updates["LLM_PROVIDER"] = "gemini"
    if args.qdrant_url:
        backend_updates["QDRANT_URL"] = args.qdrant_url
    if args.qdrant_key:
        backend_updates["QDRANT_API_KEY"] = args.qdrant_key
    if args.redis_url:
        backend_updates["REDIS_URL"] = args.redis_url
    if args.langfuse_public:
        backend_updates["LANGFUSE_PUBLIC_KEY"] = args.langfuse_public
    if args.langfuse_secret:
        backend_updates["LANGFUSE_SECRET_KEY"] = args.langfuse_secret
    if args.api_key:
        backend_updates["API_KEY"] = args.api_key

    if backend_updates:
        update_env_file(BACKEND_ENV, backend_updates)

    # If API_KEY changed, sync to frontend
    if args.api_key:
        frontend_updates = {"NEXT_PUBLIC_API_KEY": args.api_key}
        update_env_file(FRONTEND_ENV, frontend_updates)


if __name__ == "__main__":
    main()
