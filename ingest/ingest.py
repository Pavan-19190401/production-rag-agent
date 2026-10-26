import os
import sys
import json
import uuid
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from fastembed import TextEmbedding

from app.config import settings, BASE_DIR
from ingest.loaders import load_directory, load_file, calculate_file_hash, LoadedDocument
from ingest.chunking import ChunkingEngine, TextChunk

MANIFEST_PATH = BASE_DIR / "data" / "processed" / "ingest_manifest.json"
BM25_STORE_PATH = BASE_DIR / "data" / "processed" / "bm25_index.pkl"

_cached_qdrant_client: Optional[QdrantClient] = None
_cached_embed_model: Optional[TextEmbedding] = None


def get_embed_model() -> TextEmbedding:
    """Instantiates and caches a singleton TextEmbedding model."""
    global _cached_embed_model
    if _cached_embed_model is None:
        _cached_embed_model = TextEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            threads=1
        )
    return _cached_embed_model


def get_qdrant_client() -> QdrantClient:
    """Instantiates and caches a singleton Qdrant client based on configuration."""
    global _cached_qdrant_client
    if _cached_qdrant_client is not None:
        return _cached_qdrant_client

    url = settings.QDRANT_URL
    api_key = settings.QDRANT_API_KEY or None

    if url.startswith("http://") or url.startswith("https://"):
        client = QdrantClient(url=url, api_key=api_key)
    elif url == ":memory:":
        client = QdrantClient(":memory:")
    else:
        # File-system persistent path
        storage_path = Path(url)
        storage_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(storage_path))

    _cached_qdrant_client = client
    return _cached_qdrant_client


def init_qdrant_collection(client: QdrantClient, vector_size: int = 384) -> None:
    """Ensures the target Qdrant collection exists with cosine distance."""
    collection_name = settings.QDRANT_COLLECTION
    collections = [c.name for c in client.get_collections().collections]

    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE
            )
        )
        print(f"Created Qdrant collection: '{collection_name}' (vector size {vector_size})")


def load_manifest() -> Dict[str, str]:
    """Loads previous file content hashes from manifest."""
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest: Dict[str, str]) -> None:
    """Saves updated file hashes to manifest."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def deterministic_uuid(val: str) -> str:
    """Generates a stable UUID5 based on a string chunk ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, val))


def run_ingestion(
    input_dir: Optional[Path] = None,
    target_file: Optional[Path] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    strategy: str = "recursive",
    force: bool = False
) -> Dict[str, Any]:
    """
    Executes idempotent document ingestion:
    1. Scans raw directory for documents (or targets single file if specified)
    2. Compares SHA-256 hashes against manifest
    3. Chunks modified/new documents
    4. Computes BGE dense embeddings using cached singleton
    5. Upserts points into Qdrant
    6. Updates manifest and BM25 index corpus
    """
    raw_dir = input_dir or Path(settings.DATA_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)

    manifest = load_manifest() if not force else {}
    files_to_process: List[Path] = []

    if target_file and target_file.is_file():
        files_to_process = [target_file]
    else:
        supported_exts = {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}
        for f in raw_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in supported_exts:
                file_hash = calculate_file_hash(f)
                rel_path = str(f.relative_to(raw_dir))
                if manifest.get(rel_path) != file_hash:
                    files_to_process.append(f)

    if not files_to_process and not force:
        print("Idempotent check: All documents are up-to-date. Nothing to ingest.")
        return {"status": "skipped", "processed_files": 0, "chunks_indexed": 0}

    print(f"Discovered {len(files_to_process)} document(s) requiring indexing.")

    # 1. Load Documents
    loaded_docs: List[LoadedDocument] = []
    new_manifest_entries: Dict[str, str] = {}
    for f in files_to_process:
        docs = load_file(f)
        loaded_docs.extend(docs)
        try:
            rel_path = str(f.relative_to(raw_dir))
        except ValueError:
            rel_path = f.name
        new_manifest_entries[rel_path] = calculate_file_hash(f)

    if not loaded_docs:
        print("No readable text extracted from files.")
        return {"status": "empty", "processed_files": len(files_to_process), "chunks_indexed": 0}

    # 2. Chunk Documents
    chunker = ChunkingEngine(chunk_size=chunk_size, chunk_overlap=chunk_overlap, strategy=strategy)
    chunks: List[TextChunk] = chunker.chunk_documents(loaded_docs)
    print(f"Generated {len(chunks)} text chunks (strategy: '{strategy}', size: {chunk_size}, overlap: {chunk_overlap}).")

    if not chunks:
        return {"status": "no_chunks", "processed_files": len(files_to_process), "chunks_indexed": 0}

    # 3. Generate Embeddings using cached BGE model (low-memory batched inference)
    print(f"Computing embeddings with '{settings.EMBEDDING_MODEL}'...")
    embed_model = get_embed_model()
    texts = [c.content for c in chunks]
    embeddings = list(embed_model.embed(texts, batch_size=16))
    vector_dim = len(embeddings[0])

    # 4. Upsert to Qdrant
    client = get_qdrant_client()
    init_qdrant_collection(client, vector_size=vector_dim)

    points: List[qmodels.PointStruct] = []
    all_chunks_dump: List[Dict[str, Any]] = []

    for chunk, vector in zip(chunks, embeddings):
        p_id = deterministic_uuid(chunk.chunk_id)
        payload = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.content,
            "token_count": chunk.token_count,
            **chunk.metadata
        }
        points.append(qmodels.PointStruct(
            id=p_id,
            vector=vector.tolist(),
            payload=payload
        ))
        all_chunks_dump.append(payload)

    # Batch upsert
    batch_size = 64
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=batch
        )

    print(f"Successfully upserted {len(points)} points to Qdrant collection '{settings.QDRANT_COLLECTION}'.")

    # 5. Cache chunk corpus for BM25 and offline evaluation
    chunks_store_path = BASE_DIR / "data" / "processed" / "chunks.json"
    existing_chunks = []
    if chunks_store_path.exists() and not force:
        try:
            with open(chunks_store_path, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
        except Exception:
            existing_chunks = []

    # Merge by chunk_id
    chunk_map = {c["chunk_id"]: c for c in existing_chunks}
    for c in all_chunks_dump:
        chunk_map[c["chunk_id"]] = c

    with open(chunks_store_path, "w", encoding="utf-8") as f:
        json.dump(list(chunk_map.values()), f, indent=2)

    # 6. Update manifest
    manifest.update(new_manifest_entries)
    save_manifest(manifest)

    return {
        "status": "success",
        "processed_files": len(files_to_process),
        "chunks_indexed": len(points),
        "vector_dim": vector_dim
    }


def main():
    parser = argparse.ArgumentParser(description="Production RAG Ingestion Pipeline")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing raw documents")
    parser.add_argument("--chunk-size", type=int, default=settings.CHUNK_SIZE, help="Chunk token size")
    parser.add_argument("--overlap", type=int, default=settings.CHUNK_OVERLAP, help="Chunk overlap")
    parser.add_argument("--strategy", type=str, default=settings.CHUNK_STRATEGY, help="fixed, recursive, or heading")
    parser.add_argument("--force", action="store_true", help="Force re-indexing, bypassing manifest")
    args = parser.parse_args()

    input_path = Path(args.dir) if args.dir else None
    result = run_ingestion(
        input_dir=input_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        strategy=args.strategy,
        force=args.force
    )
    print(f"Ingestion completed: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
