"""PS36 - Research Paper Q&A (Streamlit UI).

Run:  streamlit run app.py
"""
import hashlib
import io
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai

from rag import PRESET_QUESTIONS, PaperIndex, answer, chunk_pages, extract_pages

load_dotenv()
st.set_page_config(page_title="Research Paper Q&A", page_icon="📄", layout="wide")


@st.cache_resource
def get_client(api_key: str):
    return genai.Client(api_key=api_key)


def build_index(client, file):
    pages = extract_pages(file)
    if not pages:
        return None, 0
    chunks = chunk_pages(pages)
    return PaperIndex(client, chunks), len(pages)


# ---------- sidebar ----------
with st.sidebar:
    st.header("Setup")
    api_key = os.getenv("GEMINI_API_KEY") or st.text_input("Gemini API key", type="password")
    uploaded = st.file_uploader("Upload a research paper (PDF)", type="pdf")
    samples = sorted(Path(__file__).parent.joinpath("samples").glob("*.pdf"))
    if samples and not uploaded:
        choice = st.selectbox("...or pick a sample paper", ["-"] + [p.name for p in samples])
        if choice != "-":
            path = Path(__file__).parent / "samples" / choice
            uploaded = io.BytesIO(path.read_bytes())
            uploaded.name = choice
    st.caption("Model: Gemini 2.5 Flash · Embeddings: gemini-embedding-001 · Vector store: FAISS")
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

# rebuild the index only when a new file is uploaded
if st.session_state.get("file_id") != file_id:
    with st.spinner("Reading and indexing the paper..."):
        try:
            index, n_pages = build_index(client, uploaded)
        except Exception as e:
            st.error(f"Indexing failed: {e}")
            st.stop()
    if index is None:
        st.error("No text found in this PDF. It may be a scanned image; try a text-based PDF.")
        st.stop()
    st.session_state.update(file_id=file_id, index=index, n_pages=n_pages, history=[])

index = st.session_state.index
st.success(f"Indexed **{uploaded.name}**: {st.session_state.n_pages} pages, {len(index.chunks)} chunks.")

# ---------- preset buttons ----------
st.subheader("Quick questions")
question = None
cols = st.columns(len(PRESET_QUESTIONS))
for col, (label, q) in zip(cols, PRESET_QUESTIONS.items()):
    if col.button(label, use_container_width=True):
        question = q

typed = st.chat_input("Ask anything about the paper...")
if typed:
    question = typed

if question:
    with st.spinner("Searching the paper and writing an answer..."):
        try:
            reply, hits = answer(client, index, question)
        except Exception as e:
            reply, hits = f"Error from the LLM API: {e}", []
    st.session_state.history.append((question, reply, hits))

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
