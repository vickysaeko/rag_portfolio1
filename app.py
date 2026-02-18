import os
import re
import numpy as np
import streamlit as st
from pypdf import PdfReader
import faiss
from openai import OpenAI
from pathlib import Path

# ========= 設定 =========
EMBED_MODEL = "text-embedding-3-small"   # 多言語性能/コスト良いEmbedding :contentReference[oaicite:2]{index=2}
CHAT_MODEL  = "gpt-4o-mini"             # 軽量で実用的 :contentReference[oaicite:3]{index=3}

TOP_K = 3
CHUNK_SIZE = 800       # 文字数ベース（まずはこれでOK）
CHUNK_OVERLAP = 150
SCORE_THRESHOLD = 0.78 # ざっくり初期値。後で調整推奨

ESCALATION_TEXT = "管理部門へお問い合わせください。"

# ========= ユーティリティ =========
def clean_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_pdfs_from_folder(folder_path):
    folder = Path(folder_path)
    pdf_paths = sorted(folder.glob("*.pdf"))

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
                    "text": text
                })

    return pages



def chunk_text(pages: list[dict], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[dict]:
    """
    ページ単位テキストをチャンク化
    戻り値: [{"chunk": "...", "page": n}, ...]
    """
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

def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm

def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    # OpenAI Embeddings
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    vecs = l2_normalize(vecs)  # 正規化して inner product == cosine 相当にする
    return vecs

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP（正規化前提でcosine相当） :contentReference[oaicite:4]{index=4}
    index.add(vectors)
    return index

def ensure_index(client, folder_path):
    if st.session_state.index is not None:
        return  # すでにロード済み

    if not Path(folder_path).exists():
        st.error("フォルダが存在しません")
        st.stop()

    # 既存インデックス確認
    loaded_index, loaded_chunks, _ = load_index_and_metadata()
    new_snapshot = compute_folder_snapshot(folder_path)

    if loaded_index is not None and not snapshot_changed(new_snapshot):
        st.session_state.index = loaded_index
        st.session_state.chunks = loaded_chunks
        st.info("インデックスをロードしました")
    else:
        with st.spinner("インデックス作成中...（初回のみ時間かかります）"):
            pages = load_pdfs_from_folder(folder_path)

            if not pages:
                st.error("PDFが見つかりません")
                st.stop()

            chunks = chunk_text(pages)
            vecs = embed_texts(client, [c["chunk"] for c in chunks])
            index = build_faiss_index(vecs)

            save_index_and_metadata(index, chunks, new_snapshot)

            st.session_state.index = index
            st.session_state.chunks = chunks

        st.success("インデックス作成完了（次回から高速）")


def format_context(retrieved: list[dict]) -> str:
    """
    retrieved: [{"page":..., "chunk":..., "score":...}, ...]
    """
    parts = []
    for i, r in enumerate(retrieved, start=1):
        parts.append(
            f"[参考{i}] ({r.get('source','')}, p.{r['page']}, score={r['score']:.3f})\n{r['chunk']}"
        )
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
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

import json
from datetime import datetime

INDEX_DIR = Path("./data/index")   # 保存先（好きに変えてOK）
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
SNAPSHOT_PATH = INDEX_DIR / "snapshot.json"

def compute_folder_snapshot(folder_path: str) -> dict:
    """
    フォルダ内PDFの状態（ファイル名/サイズ/更新時刻）を記録して、
    変更検知に使うスナップショットを作る。
    """
    folder = Path(folder_path)
    pdfs = sorted(folder.glob("*.pdf"))
    items = []
    for p in pdfs:
        stat = p.stat()
        items.append({
            "name": p.name,
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
        })
    return {
        "folder": str(folder.resolve()),
        "files": items,
        "generated_at": datetime.now().isoformat(timespec="seconds")
    }

def snapshot_changed(new_snapshot: dict) -> bool:
    """
    保存済みsnapshotと比較して、変わってたらTrue
    """
    if not SNAPSHOT_PATH.exists():
        return True
    old = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    # generated_at は比較対象外
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


# ========= Streamlit UI =========
st.set_page_config(page_title="社内PDF RAG QA", layout="wide")
st.title("社内マニュアルQA（PDF RAG / 最小構成）")

if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    st.warning("環境変数 OPENAI_API_KEY が未設定です。設定してから起動してください。")

client = OpenAI(api_key=api_key) if api_key else None

# 固定パス（必要ならここを書き換えてください）
folder_path = str(Path("data/pdfs").resolve())

has_client = client is not None
has_folder = Path(folder_path).exists()

if not has_client:
    st.warning("OPENAI_API_KEY が必要です。")

if not has_folder:
    st.warning("指定したフォルダが存在しません。パスを確認してください。")

# 起動時に自動でインデックスを準備（可能な場合のみ）
if has_client and has_folder:
    with st.spinner("インデックス準備中..."):
        ensure_index(client, folder_path)


st.divider()
st.header("質問")

question = st.text_input("質問を入力（例：交通費精算の締め日は？）", value="")
ask_btn = st.button("回答する")

if ask_btn:
    if not has_client:
        st.error("OPENAI_API_KEY が必要です。")
    elif not has_folder:
        st.error("指定したフォルダが存在しません。パスを確認してください。")
    elif not question.strip():
        st.warning("質問を入力してください。")
    else:
        ensure_index(client, folder_path)
        # 検索
        q_vec = embed_texts(client, [question])
        scores, ids = st.session_state.index.search(q_vec, TOP_K)  # scores: (1, k)
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


        # 判定：空 or スコア低い
        if (not retrieved) or (max(r["score"] for r in retrieved) < SCORE_THRESHOLD):
            st.subheader("回答")
            st.write(ESCALATION_TEXT)
            st.caption("（該当箇所が見つからない/確度が低いためエスカレーション）")
        else:
            context = format_context(retrieved)

            # 生成（根拠限定）
            with st.spinner("回答生成中..."):
                ans = answer_with_context(client, question, context)

            st.subheader("回答")
            st.write(ans)

            st.subheader("参照（検索で当たった箇所）")
            for r in retrieved:
                with st.expander(f"{r.get('source','')} / p.{r['page']} / score={r['score']:.3f}"):
                    st.write(r["chunk"])
