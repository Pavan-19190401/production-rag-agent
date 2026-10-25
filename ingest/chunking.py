import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from ingest.loaders import LoadedDocument

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        # Fallback approximation: 1 token ≈ 4 characters
        return max(1, len(text) // 4)


@dataclass
class TextChunk:
    """Represents a discrete indexed chunk ready for embedding and retrieval."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self):
        if self.token_count == 0 and self.content:
            self.token_count = count_tokens(self.content)


class ChunkingEngine:
    """
    Modular chunking engine supporting:
    - Fixed-size chunking
    - Recursive character / paragraph / sentence chunking
    - Structure-aware (heading-prefixed) chunking
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "recursive"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy.lower()

    def chunk_document(self, doc: LoadedDocument) -> List[TextChunk]:
        """Splits a single LoadedDocument into chunks based on configured strategy."""
        if not doc.content.strip():
            return []

        if self.strategy == "fixed":
            return self._chunk_fixed(doc)
        elif self.strategy == "heading" or self.strategy == "heading_aware":
            return self._chunk_heading_aware(doc)
        else:  # default to recursive
            return self._chunk_recursive(doc)

    def chunk_documents(self, docs: List[LoadedDocument]) -> List[TextChunk]:
        """Processes multiple documents into a flat list of chunks."""
        all_chunks: List[TextChunk] = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def _chunk_fixed(self, doc: LoadedDocument) -> List[TextChunk]:
        """Fixed-size chunking by token count with sliding overlap."""
        words = doc.content.split()
        if not words:
            return []

        chunks: List[TextChunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        source = doc.metadata.get("source", "doc")
        page = doc.metadata.get("page", 1)
        section = re.sub(r"[^\w\-]", "_", str(doc.metadata.get("section", "sec"))).strip("_")[:24]

        for i in range(0, len(words), step):
            slice_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(slice_words).strip()
            if not chunk_text:
                continue

            chunk_idx = len(chunks)
            chunk_id = f"{source}_p{page}_{section}_c{chunk_idx}"
            meta = dict(doc.metadata)
            meta.update({
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "strategy": "fixed",
                "char_length": len(chunk_text)
            })

            chunks.append(TextChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata=meta
            ))

            if i + self.chunk_size >= len(words):
                break

        return chunks

    def _chunk_recursive(self, doc: LoadedDocument) -> List[TextChunk]:
        """
        Recursive character chunking splitting on paragraphs, then sentences, then words.
        Ensures chunks do not exceed chunk_size while maintaining semantic cohesion.
        """
        text = doc.content
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        raw_pieces = self._split_recursive(text, separators, self.chunk_size)

        # Merge pieces into chunks up to chunk_size with overlap
        chunks: List[TextChunk] = []
        source = doc.metadata.get("source", "doc")
        page = doc.metadata.get("page", 1)
        section = re.sub(r"[^\w\-]", "_", str(doc.metadata.get("section", "sec"))).strip("_")[:24]

        accumulated_tokens = 0
        current_pieces: List[str] = []

        for piece in raw_pieces:
            piece_tokens = count_tokens(piece)
            if accumulated_tokens + piece_tokens > self.chunk_size and current_pieces:
                chunk_text = " ".join(current_pieces).strip()
                if chunk_text:
                    chunk_idx = len(chunks)
                    chunk_id = f"{source}_p{page}_{section}_c{chunk_idx}"
                    meta = dict(doc.metadata)
                    meta.update({
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_idx,
                        "strategy": "recursive",
                        "char_length": len(chunk_text)
                    })
                    chunks.append(TextChunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        metadata=meta
                    ))

                # Slide window for overlap
                overlap_tokens = 0
                overlap_pieces: List[str] = []
                for p in reversed(current_pieces):
                    pt = count_tokens(p)
                    if overlap_tokens + pt <= self.chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_tokens += pt
                    else:
                        break
                current_pieces = overlap_pieces
                accumulated_tokens = overlap_tokens

            current_pieces.append(piece)
            accumulated_tokens += piece_tokens

        # Trailing chunk
        if current_pieces:
            chunk_text = " ".join(current_pieces).strip()
            if chunk_text:
                chunk_idx = len(chunks)
                chunk_id = f"{source}_p{page}_{section}_c{chunk_idx}"
                meta = dict(doc.metadata)
                meta.update({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "strategy": "recursive",
                    "char_length": len(chunk_text)
                })
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    metadata=meta
                ))

        return chunks

    def _split_recursive(self, text: str, separators: List[str], max_size: int) -> List[str]:
        """Internal recursive helper function."""
        if not separators or count_tokens(text) <= max_size:
            return [text] if text.strip() else []

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep:
            splits = text.split(sep)
        else:
            splits = list(text)

        result: List[str] = []
        for s in splits:
            s_clean = s.strip()
            if not s_clean:
                continue
            if count_tokens(s_clean) <= max_size:
                result.append(s_clean)
            else:
                result.extend(self._split_recursive(s_clean, remaining_seps, max_size))

        return result

    def _chunk_heading_aware(self, doc: LoadedDocument) -> List[TextChunk]:
        """
        Structure-aware chunking:
        Prefixes each chunk with section heading metadata [Section: ...]
        so retrieved fragments retain their parent context.
        """
        base_chunks = self._chunk_recursive(doc)
        section = doc.metadata.get("section", "")
        source = doc.metadata.get("source", "")

        for chunk in base_chunks:
            if section and not chunk.content.startswith(f"[{section}]"):
                prefix = f"[{source} > {section}]\n"
                chunk.content = prefix + chunk.content
                chunk.token_count = count_tokens(chunk.content)
                chunk.metadata["strategy"] = "heading_aware"

        return base_chunks
