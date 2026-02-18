import os
import json
import re
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# .env を読み込む（ローカル実行向け）
load_dotenv()

# ========= 設定 =========
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
TOP_K = 8
SCORE_THRESHOLD = 0.60
ESCALATION_TEXT = "管理部門へお問い合わせください。"

# 保存先（build_index.py が作るやつ）
INDEX_DIR = Path("data/index")
FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm


def embed_query(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array([resp.data[0].embedding], dtype=np.float32)
    return l2_normalize(vec)


def format_context(retrieved: list[dict]) -> str:
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


def load_index():
    if not (FAISS_PATH.exists() and CHUNKS_PATH.exists()):
        return None, None
    index = faiss.read_index(str(FAISS_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return index, chunks


# ========= Streamlit UI =========
st.set_page_config(page_title="社内マニュアルQA", layout="wide")
st.title("社内マニュアルQA（PDF RAG）")

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    st.error("OPENAI_API_KEY が未設定です。env / 環境変数を設定してから起動してください。")
    st.stop()

client = OpenAI(api_key=api_key)

index, chunks = load_index()
if index is None:
    st.error("インデックスがありません。先に `python build_index.py` を実行してください。")
    st.stop()

st.caption(f"Loaded: {len(chunks)} chunks")

st.divider()
question = st.text_input("質問を入力（例：交通費精算の締め日は？）", value="")
ask_btn = st.button("回答する")

if ask_btn:
    if not question.strip():
        st.warning("質問を入力してください。")
        st.stop()

    q_vec = embed_query(client, question)
    scores, ids = index.search(q_vec, TOP_K)

    scores = scores[0].tolist()
    ids = ids[0].tolist()

    retrieved = []
    for s, idx in zip(scores, ids):
        if idx == -1:
            continue
        item = chunks[idx]
        retrieved.append({
            "score": float(s),
            "page": item["page"],
            "source": item.get("source", ""),
            "chunk": item["chunk"],
        })

    if (not retrieved) or (max(r["score"] for r in retrieved) < SCORE_THRESHOLD):
        st.subheader("回答")
        st.write(ESCALATION_TEXT)
        st.caption("（該当箇所が見つからない/確度が低いためエスカレーション）")
    else:
        context = format_context(retrieved)
        with st.spinner("回答生成中..."):
            ans = answer_with_context(client, question, context)

        st.subheader("回答")
        st.write(ans)

        st.subheader("参照（検索で当たった箇所）")
        for r in retrieved:
            with st.expander(f"{r.get('source','')} / p.{r['page']} / score={r['score']:.3f}"):
                st.write(r["chunk"])
