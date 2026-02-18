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
CHUNK_SIZE = 2000        # ← 少し大きめ（高速化）
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 128   # ← 超重要（速さ安定）

PDF_FOLDER = Path("data/pdfs")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
SNAPSHOT_PATH = INDEX_DIR / "snapshot.json"

# ========= ユーティリティ =========
def clean_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load_pdfs(folder_path: Path):
    pdf_paths = sorted(folder_path.glob("*.pdf"))
    pages = []

    for pdf_path in pdf_paths:
        print(f"loading: {pdf_path.name}")

        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = clean_text(text)

            if text:
                pages.append({
                    "source": pdf_path.name,
                    "page": i,
                    "text": text
                })

    return pages

def chunk_text(pages):
    chunks = []

    for p in pages:
        text = p["text"]
        start = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end]

            if chunk.strip():
                chunks.append({
                    "chunk": chunk,
                    "page": p["page"],
                    "source": p["source"]
                })

            start = end - CHUNK_OVERLAP
            if start < 0:
                start = 0

    return chunks

def l2_normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm

def embed_texts_batched(client, texts):
    all_vecs = []
    total = len(texts)

    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[i:i+EMBED_BATCH_SIZE]

        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )

        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        all_vecs.append(vecs)

        print(f"embedding: {min(i+EMBED_BATCH_SIZE, total)}/{total}")

    vecs = np.vstack(all_vecs)
    return l2_normalize(vecs)

def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def compute_snapshot(folder_path):
    pdfs = sorted(folder_path.glob("*.pdf"))
    items = []

    for p in pdfs:
        stat = p.stat()
        items.append({
            "name": p.name,
            "size": stat.st_size,
            "mtime": int(stat.st_mtime)
        })

    return {
        "folder": str(folder_path.resolve()),
        "files": items,
        "generated_at": datetime.now().isoformat(timespec="seconds")
    }

def snapshot_equal(a, b):
    return (
        a.get("folder") == b.get("folder") and
        a.get("files") == b.get("files")
    )

# ========= メイン =========
def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY を設定してください")

    if not PDF_FOLDER.exists():
        raise SystemExit(f"PDFフォルダがありません: {PDF_FOLDER}")

    new_snapshot = compute_snapshot(PDF_FOLDER)

    # 変更がなければスキップ
    if FAISS_PATH.exists() and SNAPSHOT_PATH.exists():
        old_snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if snapshot_equal(old_snapshot, new_snapshot):
            print("index up-to-date（変更なし）")
            return

    print("=== PDF読み込み ===")
    pages = load_pdfs(PDF_FOLDER)

    if not pages:
        raise SystemExit("PDFが空 or テキスト抽出失敗")

    print(f"pages: {len(pages)}")

    print("=== chunk分割 ===")
    chunks = chunk_text(pages)
    print(f"chunks: {len(chunks)}")

    client = OpenAI(api_key=api_key)

    print("=== embedding ===")
    texts = [c["chunk"] for c in chunks]
    vecs = embed_texts_batched(client, texts)

    print("=== FAISS構築 ===")
    index = build_faiss_index(vecs)

    print("=== 保存 ===")
    faiss.write_index(index, str(FAISS_PATH))
    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    SNAPSHOT_PATH.write_text(
        json.dumps(new_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("完了 🎉")

if __name__ == "__main__":
    main()
