import os
import re
import json
import hashlib
import numpy as np
import streamlit as st
from pypdf import PdfReader
import faiss
from openai import OpenAI
from pathlib import Path
from datetime import datetime

# ========= 設定 =========
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

TOP_K = 3
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
SCORE_THRESHOLD = 0.78

ESCALATION_TEXT = "管理部門へお問い合わせください。"

# Embeddingのバッチ（大きすぎると失敗しやすいので程よく）
EMBED_BATCH_SIZE = 128

# ========= 保存先 =========
INDEX_DIR = Path("./data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
SNAPSHOT_PATH = INDEX_DIR / "snapshot.json"

# ========= ユーティリティ =========
def clean_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def list_pdfs_recursive(root: str) -> list[Path]:
    root = Path(root)
    return sorted([p for p in root.rglob("*.pdf") if p.is_file()])

def load_pdfs_from_folder(folder_path: str) -> list[dict]:
    """再帰でPDFを読み込み。戻り値: [{"source": "...", "page": 1, "text": "..."}]"""
    pdf_paths = list_pdfs_recursive(folder_path)

    pages = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = clean_text(text)
            if text:
                # sourceは相対パスにしておくと同名PDFでも衝突しにくい
                rel = str(pdf_path.relative_to(Path(folder_path)))
                pages.append({"source": rel, "page": i, "text": text})
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
                    "source": p.get("source", "")
                })
            start = end - overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break
    return chunks

def make_chunk_id(source: str, page: int, chunk: str) -> str:
    """チャンクの同一性判定用ID（差分Embeddingの鍵）"""
    key = f"{source}|{page}|{chunk}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()

def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm

def embed_texts_batched(client: OpenAI, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """Embeddingをバッチで投げて安定化"""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        all_vecs.append(vecs)
    vecs = np.vstack(all_vecs) if all_vecs else np.zeros((0, 1536), dtype=np.float32)  # 次元はモデル依存だが未使用時対策
    return l2_normalize(vecs)

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # 正規化済みならcosine相当
    index.add(vectors)
    return index

def compute_folder_snapshot(folder_path: str) -> dict:
    """再帰でPDFの状態を記録（追加/変更/削除の検知用）"""
    pdfs = list_pdfs_recursive(folder_path)
    items = []
    for p in pdfs:
        stat = p.stat()
        rel = str(p.relative_to(Path(folder_path)))
        items.append({"name": rel, "size": stat.st_size, "mtime": int(stat.st_mtime)})
    return {
        "folder": str(Path(folder_path).resolve()),
        "files": items,
        "generated_at": datetime.now().isoformat(timespec="seconds")
    }

def diff_snapshot(old_snap: dict | None, new_snap: dict) -> dict:
    """追加/削除/変更を判定"""
    if not old_snap:
        return {"added": set(f["name"] for f in new_snap["files"]), "removed": set(), "modified": set()}

    old_map = {f["name"]: (f["size"], f["mtime"]) for f in old_snap.get("files", [])}
    new_map = {f["name"]: (f["size"], f["mtime"]) for f in new_snap.get("files", [])}

    old_names = set(old_map.keys())
    new_names = set(new_map.keys())

    added = new_names - old_names
    removed = old_names - new_names
    modified = {name for name in (old_names & new_names) if old_map[name] != new_map[name]}

    return {"added": added, "removed": removed, "modified": modified}

def snapshot_changed(new_snapshot: dict) -> bool:
    if not SNAPSHOT_PATH.exists():
        return True
    old = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    old_cmp = {"folder": old.get("folder"), "files": old.get("files", [])}
    new_cmp = {"folder": new_snapshot.get("folder"), "files": new_snapshot.get("files", [])}
    return old_cmp != new_cmp

def save_index_and_metadata(index, chunks: list[dict], snapshot: dict):
    faiss.write_index(index, str(FAISS_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

def load_index_and_metadata():
    if not (FAISS_PATH.exists() and CHUNKS_PATH.exists() and SNAPSHOT_PATH.exists()):
        return None, None, None
    index = faiss.read_index(str(FAISS_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    return index, chunks, snapshot

def ensure_index(client, folder_path, allow_build: bool = True):
    """差分更新：追加PDFだけEmbedding→FAISS追記。変更/削除があれば全再構築。"""
    if st.session_state.index is not None:
        return

    if not Path(folder_path).exists():
        st.error("フォルダが存在しません")
        st.stop()

    loaded_index, loaded_chunks, old_snapshot = load_index_and_metadata()
    new_snapshot = compute_folder_snapshot(folder_path)

    # 保存物がないならフルビルド
    if loaded_index is None or loaded_chunks is None or old_snapshot is None:
        if not allow_build:
            st.warning("インデックス未作成です。ALLOW_INDEX_BUILD=true で起動するか、別途ビルドしてください。")
            return
        with st.spinner("インデックス初回作成中..."):
            pages = load_pdfs_from_folder(folder_path)
            chunks = chunk_text(pages)
            # chunk_id付与
            for c in chunks:
                c["id"] = make_chunk_id(c.get("source",""), c["page"], c["chunk"])
            vecs = embed_texts_batched(client, [c["chunk"] for c in chunks])
            index = build_faiss_index(vecs)
            save_index_and_metadata(index, chunks, new_snapshot)

            st.session_state.index = index
            st.session_state.chunks = chunks
        st.success("インデックス作成完了（次回から高速）")
        return

    # 差分判定
    d = diff_snapshot(old_snapshot, new_snapshot)

    # 変更/削除があると整合性が崩れやすいので安全に全再構築
    if d["removed"] or d["modified"]:
        if not allow_build:
            st.warning("PDFが変更/削除されています。再構築が必要ですが禁止設定です。")
            return
        with st.spinner("PDF変更検知：安全のため全再構築中..."):
            pages = load_pdfs_from_folder(folder_path)
            chunks = chunk_text(pages)
            for c in chunks:
                c["id"] = make_chunk_id(c.get("source",""), c["page"], c["chunk"])
            vecs = embed_texts_batched(client, [c["chunk"] for c in chunks])
            index = build_faiss_index(vecs)
            save_index_and_metadata(index, chunks, new_snapshot)

            st.session_state.index = index
            st.session_state.chunks = chunks
        st.success("再構築完了")
        return

    # 追加だけなら高速に追記
    if d["added"]:
        with st.spinner(f"追加PDF {len(d['added'])}件を差分更新中..."):
            existing_ids = {c.get("id") for c in loaded_chunks if c.get("id")}
            # 追加ファイルだけ読む（最小化）
            added_pages = []
            for rel_name in sorted(d["added"]):
                pdf_path = Path(folder_path) / rel_name
                reader = PdfReader(str(pdf_path))
                for i, page in enumerate(reader.pages, start=1):
                    text = clean_text(page.extract_text() or "")
                    if text:
                        added_pages.append({"source": rel_name, "page": i, "text": text})

            new_chunks = chunk_text(added_pages)
            # chunk_id付与＋既存重複を弾く
            filtered = []
            for c in new_chunks:
                cid = make_chunk_id(c.get("source",""), c["page"], c["chunk"])
                c["id"] = cid
                if cid not in existing_ids:
                    filtered.append(c)

            if filtered:
                vecs = embed_texts_batched(client, [c["chunk"] for c in filtered])
                loaded_index.add(vecs)
                loaded_chunks.extend(filtered)
                save_index_and_metadata(loaded_index, loaded_chunks, new_snapshot)

        st.success("差分更新完了（追加分だけEmbedding）")

    # そのままロード
    st.session_state.index = loaded_index
    st.session_state.chunks = loaded_chunks

def format_context(retrieved: list[dict]) -> str:
    parts = []
    for i, r in enumerate(retrieved, start=1):
        parts.append(f"[参考{i}] ({r.get('source','')}, p.{r['page']}, score={r['score']:.3f})\n{r['chunk']}")
    return "\n\n".join(parts)

def answer_with_context(client: OpenAI, question: str, context: str) -> str:
    system = (
        "あなたは社内マニュアルQAのアシスタントです。"
        "以下の「参考情報」に書かれていることだけを根拠に日本語で回答してください。"
        "参考情報に書かれていない内容は推測して補わないでください。"
        f"参考情報だけでは回答できない場合は、必ず次の一文だけを返してください：{ESCALATION_TEXT}"
    )
    user = f"【質問】\n{question}\n\n【参考情報】\n{context}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

# ========= Streamlit UI =========
st.set_page_config(page_title="社内PDF RAG QA", layout="wide")
st.title("社内マニュアルQA（PDF RAG / 最小構成）")

if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None

folder_path = str(Path("data/pdfs").resolve())
allow_build = os.getenv("ALLOW_INDEX_BUILD", "true").lower() in ("1", "true", "yes")

has_client = client is not None
has_folder = Path(folder_path).exists()

if not has_client:
    st.warning("OPENAI_API_KEY が必要です。")
if not has_folder:
    st.warning("指定したフォルダが存在しません。パスを確認してください。")

# 起動時に準備（必要に応じて差分更新/再構築）
if has_client and has_folder:
    ensure_index(client, folder_path, allow_build=allow_build)

st.divider()
st.header("質問")

question = st.text_input("質問を入力（例：交通費精算の締め日は？）", value="")
ask_btn = st.button("回答する")

if ask_btn:
    if not has_client:
        st.error("OPENAI_API_KEY が必要です。")
        st.stop()
    if not has_folder:
        st.error("指定したフォルダが存在しません。パスを確認してください。")
        st.stop()
    if not question.strip():
        st.warning("質問を入力してください。")
        st.stop()

    ensure_index(client, folder_path, allow_build=allow_build)
    if st.session_state.index is None:
        st.error("インデックスがありません。ALLOW_INDEX_BUILD=true で起動して作成してください。")
        st.stop()

    q_vec = embed_texts_batched(client, [question], batch_size=1)
    scores, ids = st.session_state.index.search(q_vec, TOP_K)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    retrieved = []
    for s, idx in zip(scores, ids):
        if idx == -1:
            continue
        item = st.session_state.chunks[idx]
        retrieved.append({
            "score": float(s),
            "page": item["page"],
            "source": item.get("source", ""),
            "chunk": item["chunk"],
        })

    if (not retrieved) or (max(r["score"] for r in retrieved) < SCORE_THRESHOLD):
        st.subheader("回答")
        st.write(ESCALATION_TEX_
