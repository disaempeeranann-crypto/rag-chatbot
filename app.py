import io

import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# ----------------------------- ตั้งค่า -----------------------------
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # รองรับภาษาไทย
POLLINATIONS_URL = "https://text.pollinations.ai/openai"  # LLM ฟรี ไม่ต้องใช้ API key
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

SYSTEM_PROMPT = "คุณเป็นผู้ช่วย AI ที่เป็นมิตร ตอบเป็นภาษาเดียวกับผู้ใช้ กระชับและถูกต้อง"


# ----------------------------- Embedding -----------------------------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล Embedding (ครั้งแรกอาจใช้เวลาสักครู่)...")
def get_embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(EMBED_MODEL)


def embed(texts):
    model = get_embedder()
    vecs = np.array(list(model.embed(texts)), dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


# ----------------------------- อ่านเอกสาร -----------------------------
def extract_text(file) -> str:
    name = file.name.lower()
    data = file.getvalue()

    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)

    if name.endswith(".docx"):
        from docx import Document

        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    # txt, md, csv
    for enc in ("utf-8", "utf-8-sig", "cp874", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return ""


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


# ----------------------------- Knowledge base -----------------------------
def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chunks", [])  # [{"text":..., "source":...}]
    st.session_state.setdefault("emb", None)
    st.session_state.setdefault("indexed_files", set())


def add_documents(files):
    new_chunks = []
    for f in files:
        if f.name in st.session_state.indexed_files:
            continue
        text = extract_text(f)
        for c in chunk_text(text):
            new_chunks.append({"text": c, "source": f.name})
        st.session_state.indexed_files.add(f.name)

    if not new_chunks:
        return 0

    vecs = embed([c["text"] for c in new_chunks])
    st.session_state.chunks.extend(new_chunks)
    st.session_state.emb = (
        vecs if st.session_state.emb is None else np.vstack([st.session_state.emb, vecs])
    )
    return len(new_chunks)


def clear_kb():
    st.session_state.chunks = []
    st.session_state.emb = None
    st.session_state.indexed_files = set()


def retrieve(query, k, threshold):
    """คืนค่า (รายการ chunk ที่เกี่ยวข้อง, คะแนนสูงสุด)"""
    if st.session_state.emb is None or not st.session_state.chunks:
        return [], 0.0
    q = embed([query])[0]
    scores = st.session_state.emb @ q
    top = np.argsort(-scores)[:k]
    best = float(scores[top[0]])
    hits = [
        {**st.session_state.chunks[i], "score": float(scores[i])}
        for i in top
        if scores[i] >= threshold
    ]
    return hits, best


# ----------------------------- LLM (ฟรี ไม่ใช้ API key) -----------------------------
def call_free_llm(messages, timeout=60):
    resp = requests.post(
        POLLINATIONS_URL,
        json={"model": "openai", "messages": messages},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_local_llm(messages):
    """ทางเลือก: รันโมเดลเล็กในเครื่องเอง (ต้องติดตั้ง transformers + torch)"""
    pipe = get_local_pipeline()
    out = pipe(messages, max_new_tokens=400)
    return out[0]["generated_text"][-1]["content"]


@st.cache_resource(show_spinner="กำลังโหลดโมเดลในเครื่อง...")
def get_local_pipeline():
    from transformers import pipeline

    return pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")


def ask_llm(messages, backend):
    try:
        if backend == "โมเดลในเครื่อง (Qwen 0.5B)":
            return call_local_llm(messages)
        return call_free_llm(messages)
    except Exception as e:
        return f"⚠️ เรียก LLM ไม่สำเร็จ: {e}"


# ----------------------------- UI -----------------------------
init_state()

with st.sidebar:
    st.header("📚 ฐานความรู้ (RAG)")
    files = st.file_uploader(
        "อัปโหลดเอกสาร (PDF, DOCX, TXT, MD, CSV)",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
    )
    if st.button("➕ เพิ่มเข้าฐานความรู้", use_container_width=True, disabled=not files):
        with st.spinner("กำลังสร้าง index..."):
            n = add_documents(files)
        st.success(f"เพิ่ม {n} chunk สำเร็จ" if n else "ไม่มีเอกสารใหม่ให้เพิ่ม")

    st.caption(
        f"เอกสาร: {len(st.session_state.indexed_files)} ไฟล์ | "
        f"Chunks: {len(st.session_state.chunks)}"
    )
    if st.session_state.indexed_files:
        st.write("\n".join(f"- {n}" for n in sorted(st.session_state.indexed_files)))
        if st.button("🗑️ ล้างฐานความรู้", use_container_width=True):
            clear_kb()
            st.rerun()

    st.divider()
    st.header("⚙️ ตั้งค่า")
    backend = st.selectbox(
        "LLM ที่ใช้ตอบ",
        ["Pollinations (ฟรี ไม่ต้องใช้ key)", "โมเดลในเครื่อง (Qwen 0.5B)"],
    )
    top_k = st.slider("จำนวน chunk ที่ดึงมา (Top-K)", 1, 8, 4)
    threshold = st.slider(
        "เกณฑ์ความเกี่ยวข้องขั้นต่ำ", 0.0, 1.0, 0.35, 0.05,
        help="ถ้าคะแนนต่ำกว่านี้ จะถือว่าไม่พบข้อมูลในเอกสาร แล้วให้ LLM ตอบเอง",
    )
    if st.button("🧹 ล้างประวัติแชท", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 RAG Chatbot")
st.caption("ถามจากเอกสารที่คุณอัปโหลด — ถ้าไม่พบข้อมูลในเอกสาร จะให้ LLM ฟรีตอบแทน")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📄 แหล่งอ้างอิง"):
                for s in m["sources"]:
                    st.markdown(f"**{s['source']}** (score {s['score']:.2f})")
                    st.caption(s["text"][:400] + "...")

if prompt := st.chat_input("พิมพ์คำถามของคุณ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("กำลังคิด..."):
            hits, best = retrieve(prompt, top_k, threshold)

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-7:-1]
            ]

            if hits:
                context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
                sys_msg = (
                    SYSTEM_PROMPT
                    + "\n\nใช้ข้อมูลจากเอกสารด้านล่างตอบคำถาม "
                    "ถ้าเอกสารไม่ครอบคลุมบางส่วน ให้บอกอย่างชัดเจนว่าส่วนนั้นไม่มีในเอกสาร\n\n"
                    f"=== เอกสาร ===\n{context}"
                )
                mode = "📚 ตอบจากเอกสารของคุณ"
            else:
                sys_msg = SYSTEM_PROMPT
                mode = "🌐 ไม่พบข้อมูลในเอกสาร — ตอบจากความรู้ทั่วไปของ LLM"

            messages = [{"role": "system", "content": sys_msg}] + history + [
                {"role": "user", "content": prompt}
            ]
            answer = ask_llm(messages, backend)

        st.caption(mode)
        st.markdown(answer)
        if hits:
            with st.expander("📄 แหล่งอ้างอิง"):
                for s in hits:
                    st.markdown(f"**{s['source']}** (score {s['score']:.2f})")
                    st.caption(s["text"][:400] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": f"_{mode}_\n\n{answer}", "sources": hits}
    )
