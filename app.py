"""PS36 - Research Paper Q&A (Streamlit UI).

Run:  streamlit run app.py
Optional URL params for demos: ?sample=<file.pdf>&ask=<Method|Dataset|Results|Limitations|free text>
"""
import hashlib
import io
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag import PRESET_QUESTIONS, SEARCH_HINTS, PaperIndex, answer, chunk_pages, current_model, extract_pages

load_dotenv()
st.set_page_config(page_title="Research Paper Q&A", page_icon="📄", layout="wide")
SAMPLES_DIR = Path(__file__).parent / "samples"


@st.cache_resource
def get_client(api_key: str):
    # 30 s per request so an overloaded model fails fast and the next one is tried
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30000))


@st.cache_resource(show_spinner=False, max_entries=10)
def build_index(_client, file_id: str, file_bytes: bytes):
    """Cached per file hash, so re-opening the same paper never re-embeds it."""
    pages = extract_pages(io.BytesIO(file_bytes))
    if not pages:
        return None, 0
    chunks = chunk_pages(pages)
    return PaperIndex(_client, chunks), len(pages)


params = st.query_params

# ---------- sidebar ----------
with st.sidebar:
    st.header("Setup")
    api_key = os.getenv("GEMINI_API_KEY") or st.text_input("Gemini API key", type="password")
    uploaded = st.file_uploader("Upload a research paper (PDF)", type="pdf")
    samples = sorted(p.name for p in SAMPLES_DIR.glob("*.pdf"))
    if samples and not uploaded:
        options = ["-"] + samples
        default = options.index(params["sample"]) if params.get("sample") in options else 0
        choice = st.selectbox("...or pick a sample paper", options, index=default)
        if choice != "-":
            uploaded = io.BytesIO((SAMPLES_DIR / choice).read_bytes())
            uploaded.name = choice
    model_slot = st.empty()
    if st.button("Clear chat"):
        st.session_state.history = []

st.title("📄 Research Paper Q&A")
st.write("Upload a paper, then ask a question or use a preset button. Answers come only from the paper and cite page numbers.")

if not api_key:
    st.info("Add your Gemini API key in the sidebar or in a `.env` file to start.")
    st.stop()
if not uploaded:
    st.info("Upload a PDF (or pick a sample) in the sidebar to start.")
    st.stop()

client = get_client(api_key)
file_bytes = uploaded.getvalue()
file_id = hashlib.md5(file_bytes).hexdigest()

# rebuild the index only when a new file is chosen
if st.session_state.get("file_id") != file_id:
    with st.spinner("Reading and indexing the paper..."):
        try:
            index, n_pages = build_index(client, file_id, file_bytes)
        except Exception as e:
            st.error(f"Indexing failed: {e}")
            st.stop()
    if index is None:
        st.error("No text found in this PDF. It may be a scanned image; try a text-based PDF.")
        st.stop()
    st.session_state.update(file_id=file_id, index=index, n_pages=n_pages, history=[], auto_done=False)

index = st.session_state.index
st.success(f"Indexed **{uploaded.name}**: {st.session_state.n_pages} pages, {len(index.chunks)} chunks.")

# ---------- preset buttons ----------
st.subheader("Quick questions")
question, hint = None, ""
cols = st.columns(len(PRESET_QUESTIONS))
for col, (label, q) in zip(cols, PRESET_QUESTIONS.items()):
    if col.button(label, use_container_width=True):
        question, hint = q, SEARCH_HINTS[label]

typed = st.chat_input("Ask anything about the paper...")
if typed:
    question = typed

# demo mode: ?ask=... runs one question automatically per paper
if not question and params.get("ask") and not st.session_state.get("auto_done"):
    question = PRESET_QUESTIONS.get(params["ask"], params["ask"])
    hint = SEARCH_HINTS.get(params["ask"], "")
    st.session_state.auto_done = True

if question:
    with st.spinner("Searching the paper and writing an answer..."):
        try:
            reply, hits = answer(client, index, question, hint)
        except Exception as e:
            reply, hits = f"Error from the LLM API: {e}", []
    st.session_state.history.append((question, reply, hits))

model_slot.caption(f"Model: {current_model()} · Embeddings: gemini-embedding-001 · Vector store: FAISS")

# ---------- chat history (newest first) ----------
for q, reply, hits in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.markdown(reply)
        if hits:
            with st.expander("Sources used"):
                for chunk, score in hits:
                    st.markdown(f"**Page {chunk.page}** · similarity {score:.2f}")
                    st.caption(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))
