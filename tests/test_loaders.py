from pathlib import Path
import pytest
from ingest.loaders import clean_text, calculate_file_hash, load_document, load_markdown, load_html, load_pdf

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_clean_text():
    raw = "  Hello   world \u200b\n\nPage 4 of 12\n\n   Paragraph  two.   \n\n\n\nParagraph 3.  "
    cleaned = clean_text(raw)
    assert "Page 4 of 12" not in cleaned
    assert "Hello world" in cleaned
    assert "Paragraph two." in cleaned
    assert "Paragraph 3." in cleaned
    assert "\u200b" not in cleaned


def test_calculate_file_hash(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello Production RAG", encoding="utf-8")
    hash1 = calculate_file_hash(test_file)
    assert len(hash1) == 64  # SHA-256 length

    test_file.write_text("Hello Production RAG Modified", encoding="utf-8")
    hash2 = calculate_file_hash(test_file)
    assert hash1 != hash2


def test_load_markdown():
    md_file = TEST_DATA_DIR / "flash_attention.md"
    if not md_file.exists():
        pytest.skip("flash_attention.md does not exist")

    docs = load_markdown(md_file)
    assert len(docs) > 0
    sections = [d.metadata.get("section") for d in docs]
    assert any("Abstract" in s for s in sections)
    assert docs[0].metadata["doc_type"] == "markdown"
    assert "source" in docs[0].metadata


def test_load_html():
    html_file = TEST_DATA_DIR / "rag_system_architecture.html"
    if not html_file.exists():
        pytest.skip("rag_system_architecture.html does not exist")

    docs = load_html(html_file)
    assert len(docs) > 0
    content = docs[0].content
    assert "Navigation Boilerplate" not in content  # nav stripped
    assert "2026 Production RAG Systems Inc" not in content  # footer stripped
    assert "Hybrid Retrieval and Reciprocal Rank Fusion" in content
    assert docs[0].metadata["doc_type"] == "html"


def test_load_pdf():
    pdf_file = TEST_DATA_DIR / "rag_paper.pdf"
    if not pdf_file.exists():
        pytest.skip("rag_paper.pdf not present in data/raw")

    docs = load_pdf(pdf_file)
    assert len(docs) > 0
    assert docs[0].metadata["doc_type"] == "pdf"
    assert docs[0].metadata["page"] == 1
    assert "total_pages" in docs[0].metadata
    assert len(docs[0].content) > 50
