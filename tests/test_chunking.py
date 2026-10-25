import pytest
from ingest.loaders import LoadedDocument
from ingest.chunking import ChunkingEngine, TextChunk, count_tokens


def test_chunking_fixed():
    sample_text = " ".join([f"word{i}" for i in range(200)])
    doc = LoadedDocument(
        content=sample_text,
        metadata={"source": "test.txt", "page": 1, "section": "main"}
    )

    chunker = ChunkingEngine(chunk_size=50, chunk_overlap=10, strategy="fixed")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    assert chunks[0].metadata["strategy"] == "fixed"
    assert chunks[0].chunk_id.startswith("test.txt_p1_main_c0")
    assert chunks[1].chunk_id.startswith("test.txt_p1_main_c1")


def test_chunking_recursive():
    sample_text = (
        "Introduction to RAG.\n\n"
        "Retrieval-Augmented Generation combines dense vector retrieval with pre-trained seq2seq models. "
        "It achieves state-of-the-art results on open-domain question answering.\n\n"
        "Here is a second paragraph discussing BM25 and hybrid search methods."
    )
    doc = LoadedDocument(
        content=sample_text,
        metadata={"source": "rag_intro.md", "page": 1, "section": "Intro"}
    )

    chunker = ChunkingEngine(chunk_size=40, chunk_overlap=10, strategy="recursive")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.token_count > 0
        assert c.metadata["strategy"] == "recursive"


def test_chunking_heading_aware():
    sample_text = "Tiling reduces SRAM traffic significantly."
    doc = LoadedDocument(
        content=sample_text,
        metadata={"source": "flash.md", "page": 1, "section": "Tiling Algorithm"}
    )

    chunker = ChunkingEngine(chunk_size=100, chunk_overlap=10, strategy="heading")
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert "[flash.md > Tiling Algorithm]" in chunks[0].content
    assert chunks[0].metadata["strategy"] == "heading_aware"
