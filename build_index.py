import os
import json
import re
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

# ========= 設定 =========
EMBED_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1200        # ← 精度優先で少し細かく
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 64    # ← 安定性重視
MAX_TEXT_CHARS = 20000   # ← 1ページの最大文字数（暴発防止）
MAX_CHUNKS = 2000        # ← 全体の最大チャンク数（暴発防止）

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
    # 日本語の文字間スペースを詰める
    t = re.sub(r"(?<=[\u3040-\u30ff\u4e00-\u9fff])\s+(?=[\u3040-\u30ff\u4e00-\u9fff])", "", t)
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
                if len(text) > MAX_TEXT_CHARS:
                    text = text[:MAX_TEXT_CHARS]
                    print(f"truncate: {pdf_path.name} p.{i} -> {MAX_TEXT_CHARS} chars")
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
                if len(chunks) >= MAX_CHUNKS:
                    print(f"chunk limit reached: {MAX_CHUNKS}")
                    return chunks
                chunks.append({
                    "chunk": chunk,
                    "page": p["page"],
                    "source": p["source"]
                })

            if end >= len(text):
                break          # ← テキストの末尾まで処理したら抜ける
            start = end - CHUNK_OVERLAP

    return chunks


def l2_normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm

import time
from typing import List

def embed_texts_batched(client, texts: List[str], max_retries: int = 6, sleep_base: float = 1.5):
    """
    - バッチでEmbedding
    - 失敗時は指数バックオフでリトライ
    - dataのindexで順序を元に戻す（安全策）
    """
    all_vecs = []
    total = len(texts)

    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]

        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)

                # 順序保証（念のためindexで並び替え）
                data_sorted = sorted(resp.data, key=lambda x: x.index)
                vecs = np.array([d.embedding for d in data_sorted], dtype=np.float32)

                all_vecs.append(vecs)
                done = min(i + EMBED_BATCH_SIZE, total)
                print(f"embedding: {done}/{total}")
                break

            except Exception as e:
                wait = sleep_base ** attempt
                print(f"[warn] embedding failed (batch {i}-{i+len(batch)}), retry {attempt+1}/{max_retries}, wait {wait:.1f}s\n{e}")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Embedding failed permanently at batch starting {i}")

    vecs = np.vstack(all_vecs) if all_vecs else np.zeros((0, 1), dtype=np.float32)
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
    load_dotenv()
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
