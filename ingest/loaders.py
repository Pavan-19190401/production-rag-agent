import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pypdf
from bs4 import BeautifulSoup


@dataclass
class LoadedDocument:
    """Represents an ingested text document with metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """
    Cleans raw extracted text:
    - Normalizes unicode spaces and tabs
    - Strips common header/footer boilerplate and page numbering
    - Collapses excessive newlines while preserving paragraph structure
    """
    if not text:
        return ""

    # Replace non-breaking spaces and irregular whitespace
    text = re.sub(r"[\u200b\u200c\u200d\uFEFF]", "", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove common repetitive footer/header page number patterns
    text = re.sub(r"(?i)page\s+\d+\s+(of|\/)\s+\d+", "", text)
    text = re.sub(r"(?i)\b\d+\s*\|\s*page\b", "", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)  # Standalone page number lines

    # Normalize newlines: no more than 2 consecutive newlines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def calculate_file_hash(file_path: Path) -> str:
    """Computes SHA-256 of the raw file for idempotency checking."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_pdf(file_path: Path) -> List[LoadedDocument]:
    """Loads a PDF document page-by-page with page-level metadata."""
    documents: List[LoadedDocument] = []
    file_hash = calculate_file_hash(file_path)
    file_mod_time = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()

    try:
        reader = pypdf.PdfReader(str(file_path))
        num_pages = len(reader.pages)
        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = clean_text(raw_text)
            if not cleaned:
                continue

            metadata = {
                "source": file_path.name,
                "file_path": str(file_path),
                "doc_type": "pdf",
                "page": page_num,
                "total_pages": num_pages,
                "section": f"Page {page_num}",
                "date": file_mod_time,
                "file_hash": file_hash
            }
            documents.append(LoadedDocument(content=cleaned, metadata=metadata))
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")

    return documents


def load_markdown(file_path: Path) -> List[LoadedDocument]:
    """Loads a Markdown file, splitting into section documents based on headers."""
    documents: List[LoadedDocument] = []
    file_hash = calculate_file_hash(file_path)
    file_mod_time = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        # Split by top-level or second-level headers (# or ##)
        header_pattern = re.compile(r"(?m)^(#{1,3}\s+.+)$")
        parts = header_pattern.split(raw_text)

        current_section = file_path.stem
        accumulated_text = ""

        if len(parts) <= 1:
            # No headings found, load entire document as single unit
            cleaned = clean_text(raw_text)
            if cleaned:
                metadata = {
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "doc_type": "markdown",
                    "page": 1,
                    "section": current_section,
                    "date": file_mod_time,
                    "file_hash": file_hash
                }
                documents.append(LoadedDocument(content=cleaned, metadata=metadata))
            return documents

        # Iterate through matched headers and body text
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("#"):
                # If we have accumulated text from prior section, save it
                if accumulated_text.strip():
                    cleaned = clean_text(accumulated_text)
                    if cleaned:
                        metadata = {
                            "source": file_path.name,
                            "file_path": str(file_path),
                            "doc_type": "markdown",
                            "page": 1,
                            "section": current_section,
                            "date": file_mod_time,
                            "file_hash": file_hash
                        }
                        documents.append(LoadedDocument(content=cleaned, metadata=metadata))
                current_section = part.lstrip("#").strip()
                accumulated_text = part + "\n\n"
            else:
                accumulated_text += part + "\n\n"

        # Final section
        if accumulated_text.strip():
            cleaned = clean_text(accumulated_text)
            if cleaned:
                metadata = {
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "doc_type": "markdown",
                    "page": 1,
                    "section": current_section,
                    "date": file_mod_time,
                    "file_hash": file_hash
                }
                documents.append(LoadedDocument(content=cleaned, metadata=metadata))

    except Exception as e:
        print(f"Error loading Markdown {file_path}: {e}")

    return documents


def load_html(file_path: Path) -> List[LoadedDocument]:
    """Loads an HTML document, stripping nav/footer/scripts and capturing clean text."""
    documents: List[LoadedDocument] = []
    file_hash = calculate_file_hash(file_path)
    file_mod_time = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_html = f.read()

        soup = BeautifulSoup(raw_html, "html.parser")

        # Decompose non-content elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            element.decompose()

        # Extract title or main heading
        title = soup.title.string.strip() if soup.title and soup.title.string else file_path.stem
        raw_text = soup.get_text(separator="\n\n")
        cleaned = clean_text(raw_text)

        if cleaned:
            metadata = {
                "source": file_path.name,
                "file_path": str(file_path),
                "doc_type": "html",
                "page": 1,
                "section": title,
                "date": file_mod_time,
                "file_hash": file_hash
            }
            documents.append(LoadedDocument(content=cleaned, metadata=metadata))
    except Exception as e:
        print(f"Error loading HTML {file_path}: {e}")

    return documents


def load_document(file_path: Path) -> List[LoadedDocument]:
    """Factory loader dispatching based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix in [".md", ".markdown"]:
        return load_markdown(file_path)
    elif suffix in [".html", ".htm"]:
        return load_html(file_path)
    elif suffix in [".txt"]:
        return load_markdown(file_path)  # Plain text treated like simple markdown
    else:
        return []

# Alias for convenience
load_file = load_document


def load_directory(dir_path: Path) -> List[LoadedDocument]:
    """Recursively loads all supported documents from a directory."""
    all_docs: List[LoadedDocument] = []
    if not dir_path.exists():
        return all_docs

    supported_exts = {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}
    for file_path in dir_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_exts:
            docs = load_document(file_path)
            all_docs.extend(docs)
    return all_docs
