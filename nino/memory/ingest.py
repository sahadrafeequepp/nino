"""Document ingestion pipeline for note RAG with chunking, overlap, and incremental indexing."""

import hashlib
import os
from pathlib import Path
from typing import Any
import uuid

from nino.memory.db import MemoryDB
from nino.memory.vector.base import Embedder, VectorStore


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file for incremental change detection."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


from nino.security.antivirus import antivirus_scanner, AntivirusThreatDetected


def extract_file_text(path: Path) -> str:
    """Extract plain text from .md, .txt, or .pdf files after antivirus verification."""
    # 1. Deep scan file with AntivirusScanner
    scan = antivirus_scanner.scan_file(path, quarantine_on_threat=True)
    if not scan.is_clean:
        raise AntivirusThreatDetected(
            f"File '{path.name}' blocked and quarantined by Antivirus: [{scan.threat_name}] {scan.details}"
        )

    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages_text)
        except Exception as e:
            raise RuntimeError(f"Failed to extract PDF text from {path}: {e}")
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: .md, .txt, .pdf")


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[dict[str, Any]]:
    """Split text into overlapping character chunks preserving approximate boundaries."""
    chunks = []
    text_len = len(text)
    if text_len == 0:
        return []

    start = 0
    chunk_index = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # Attempt to split on newline or space near end if not at the absolute end
        if end < text_len:
            space_pos = text.rfind(" ", start, end)
            newline_pos = text.rfind("\n", start, end)
            split_pos = max(space_pos, newline_pos)
            if split_pos > start + (chunk_size // 2):
                end = split_pos

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "content": chunk_content,
                }
            )
            chunk_index += 1

        if end >= text_len:
            break
        start = max(start + 1, end - overlap)

    return chunks


class DocumentIngestionPipeline:
    """Incremental document indexing and retrieval preparation."""

    def __init__(
        self,
        memory_db: MemoryDB,
        vector_store: VectorStore,
        embedder: Embedder,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        overlap: int | None = None,
    ):
        self.memory_db = memory_db
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = overlap if overlap is not None else chunk_overlap

    async def ingest_directory(self, folder_path: str | Path) -> dict[str, int]:
        """Recursively scan directory and incrementally index .md, .txt, and .pdf files."""
        folder = Path(folder_path).resolve()
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Folder path does not exist or is not a directory: {folder}")

        supported_extensions = {".md", ".txt", ".pdf"}
        found_files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in supported_extensions]

        indexed_count = 0
        skipped_count = 0

        # Current files map: normalized string path -> Path
        current_files_map = {str(p): p for p in found_files}

        # Check existing indexed documents from DB
        with self.memory_db.get_connection() as conn:
            cur = conn.execute("SELECT id, file_path, file_hash FROM documents")
            existing_docs = {r["file_path"]: (r["id"], r["file_hash"]) for r in cur.fetchall()}

        # 1. Clean up deleted files from this folder subtree
        folder_str = str(folder)
        for old_path, (old_doc_id, _) in existing_docs.items():
            if old_path.startswith(folder_str) and old_path not in current_files_map:
                await self.delete_document(old_doc_id)

        # 2. Process existing or new files
        for file_str, file_path in current_files_map.items():
            current_hash = compute_file_hash(file_path)

            if file_str in existing_docs:
                existing_doc_id, existing_hash = existing_docs[file_str]
                if existing_hash == current_hash:
                    skipped_count += 1
                    continue
                # File changed: delete previous version before re-indexing
                await self.delete_document(existing_doc_id)

            # Index new/changed file
            await self._index_single_file(file_path, current_hash)
            indexed_count += 1

        return {"indexed": indexed_count, "skipped": skipped_count, "total_scanned": len(found_files)}

    async def _index_single_file(self, file_path: Path, file_hash: str) -> None:
        """Extract text, chunk, embed, and store document chunks in DB and vector store."""
        raw_text = extract_file_text(file_path)
        chunks = chunk_text(raw_text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            return

        doc_id = str(uuid.uuid4())
        file_path_str = str(file_path)

        with self.memory_db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, file_path, file_hash, chunk_count)
                VALUES (?, ?, ?, ?)
                """,
                (doc_id, file_path_str, file_hash, len(chunks)),
            )

        # Batch embed all chunk contents
        texts = [c["content"] for c in chunks]
        embeddings = await self.embedder.embed_batch(texts)

        chunk_records = []
        with self.memory_db.get_connection() as conn:
            for c, emb in zip(chunks, embeddings):
                chunk_id = str(uuid.uuid4())
                enc_content = self.memory_db.encrypt_text(c["content"])
                conn.execute(
                    """
                    INSERT INTO document_chunks (id, doc_id, file_path, chunk_index, start_char, end_char, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chunk_id, doc_id, file_path_str, c["chunk_index"], c["start_char"], c["end_char"], enc_content),
                )
                chunk_records.append((chunk_id, emb, c))

        # Index in vector store outside open connection block
        for chunk_id, emb, c in chunk_records:
            await self.vector_store.upsert(
                id=f"chunk_{chunk_id}",
                vector=emb,
                entity_type="document_chunk",
                entity_id=chunk_id,
                metadata={
                    "doc_id": doc_id,
                    "file_path": file_path_str,
                    "chunk_index": c["chunk_index"],
                    "start_char": c["start_char"],
                    "end_char": c["end_char"],
                    "text": c["content"],
                },
            )

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks from SQLite and vector store."""
        # Find chunk IDs to remove from vector store
        with self.memory_db.get_connection() as conn:
            cur = conn.execute("SELECT id FROM document_chunks WHERE doc_id = ?", (doc_id,))
            chunk_ids = [r["id"] for r in cur.fetchall()]
            cur_del = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            deleted = cur_del.rowcount > 0

        for cid in chunk_ids:
            await self.vector_store.delete(f"chunk_{cid}")

        return deleted
