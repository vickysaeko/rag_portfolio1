import os
import json
import re
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from pypdf import PdfReader

# ========= 設定 =========
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100

PDF_FOLDER = Path("data/pdfs")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
SNAPSHOT_PATH = INDEX_DIR / "snapshot.json"


def clean_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_pdfs_from_folder(folder_path: Path):
    pdf_paths = sorted(folder_path.glob("*.pdf"))
    pages = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = clean_text(text)
            if text:
                pages.append({
                    "source": pdf_path.name,
                    "page": i,
                    "text": text,
                })
    return pages


def chunk_text(pages: list[dict], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[dict]:
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "chunk": chunk,
                    "page": p["page"],
                    "source": p.get("source", ""),
                })
            start = end - overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break
    return chunks


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return l2_normalize(vecs)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def compute_folder_snapshot(folder_path: Path) -> dict:
    pdfs = sorted(folder_path.glob("*.pdf"))
    items = []
    for p in pdfs:
        stat = p.stat()
        items.append({
            "name": p.name,
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
        })
    return {
        "folder": str(folder_path.resolve()),
        "files": items,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_index_and_metadata(index, chunks: list[dict], snapshot: dict):
    faiss.write_index(index, str(FAISS_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY が未設定です。")

    if not PDF_FOLDER.exists():
        raise SystemExit(f"PDFフォルダが存在しません: {PDF_FOLDER}")

    pages = load_pdfs_from_folder(PDF_FOLDER)
    if not pages:
        raise SystemExit("PDFが見つかりません。")

    client = OpenAI(api_key=api_key)

    chunks = chunk_text(pages)
    vecs = embed_texts(client, [c["chunk"] for c in chunks])
    index = build_faiss_index(vecs)

    snapshot = compute_folder_snapshot(PDF_FOLDER)
    save_index_and_metadata(index, chunks, snapshot)
    print(f"done: {len(chunks)} chunks -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
