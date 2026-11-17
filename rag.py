"""RAG core for PS36 - Research Paper Q&A.

Pipeline: PDF -> page text -> overlapping chunks -> Gemini embeddings -> FAISS index
-> top-k retrieval -> grounded prompt -> Gemini answer with page citations.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import numpy as np
from pypdf import PdfReader

EMBED_MODEL = "gemini-embedding-001"
# Gemini models get retired (404) or overloaded (503/504), so try these in order
# and keep using the first one that answers. GEMINI_MODEL in .env overrides the order.
CHAT_MODELS = [m for m in [os.getenv("GEMINI_MODEL"), "gemini-3.5-flash", "gemini-3.5-flash-lite",
                           "gemini-3.6-flash", "gemini-3.1-flash-lite"] if m]
RETRYABLE = ("NOT_FOUND", "404", "UNAVAILABLE", "503", "DEADLINE_EXCEEDED", "504", "timed out", "Timeout")
_working_model: str | None = None
CHUNK_SIZE = 1200      # characters per chunk
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks
TOP_K = 8
EMBED_BATCH = 50

PRESET_QUESTIONS = {
    "Method": "What method, model or approach does this paper propose? Explain the main steps or architecture.",
    "Dataset": "Which datasets are used in this paper? Give their names, size and how they were split or prepared.",
    "Results": "What are the main results? List the key metrics and numbers and compare them with the baselines.",
    "Limitations": "What are the limitations of this work, and what future work do the authors suggest?",
}

# Extra keywords added to the retrieval query only (not shown to the LLM). Papers rarely use the
# word "dataset" for benchmarks, so a plain-question search can miss the right chunks.
SEARCH_HINTS = {
    "Method": "proposed framework architecture model pipeline algorithm training",
    "Dataset": "dataset benchmark corpus training data evaluation data samples videos tasks split collected",
    "Results": "results table accuracy success rate score improvement baseline outperforms",
    "Limitations": "limitations future work drawbacks challenges fails conclusion",
}

SYSTEM_PROMPT = """You are a research assistant helping a student write a literature review.
Answer the question using ONLY the context excerpts from the paper given below.
Rules:
1. If the context contains nothing relevant to the question, reply exactly: "Not found in the paper."
   If it contains part of the answer, give that part and say what is missing.
2. Cite the page number after each fact, like (p. 4).
3. Use short bullet points. Keep numbers, dataset names and metric names exactly as written.
4. Do not use outside knowledge and do not guess."""


@dataclass
class Chunk:
    text: str
    page: int


def extract_pages(file) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for pages that contain text."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated line breaks
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(pages: list[tuple[int, str]], size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Split each page into overlapping chunks, keeping the page number for citations."""
    chunks = []
    for page_no, text in pages:
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            # try to end the chunk on a sentence boundary
            if end < len(text):
                dot = text.rfind(". ", start + size // 2, end)
                if dot != -1:
                    end = dot + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(piece, page_no))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(client, texts: list[str], task_type: str) -> np.ndarray:
    from google.genai import types

    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        for attempt in range(4):
            try:
                result = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                break
            except Exception as e:  # free tier: ~100 embeddings/minute -> wait and retry on 429
                if ("429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e)) or attempt == 3:
                    raise
                print(f"[rag] embedding rate limit hit, waiting 30 s (attempt {attempt + 1})", flush=True)
                time.sleep(30)
        vectors.extend(e.values for e in result.embeddings)
    return _normalize(np.array(vectors, dtype="float32"))


class PaperIndex:
    """FAISS inner-product index over normalized vectors, which gives cosine similarity."""

    def __init__(self, client, chunks: list[Chunk]):
        import faiss

        self.client = client
        self.chunks = chunks
        vectors = embed_texts(client, [c.text for c in chunks], "RETRIEVAL_DOCUMENT")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        q = embed_texts(self.client, [query], "RETRIEVAL_QUERY")
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(f"[Page {c.page}]\n{c.text}" for c, _ in hits)
    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"


def answer(client, index: PaperIndex, question: str, search_hint: str = "") -> tuple[str, list[tuple[Chunk, float]]]:
    global _working_model
    hits = index.search(f"{question} {search_hint}".strip())
    prompt = build_prompt(question, hits)
    candidates = list(dict.fromkeys(([_working_model] if _working_model else []) + CHAT_MODELS))
    last_error = None
    for model in candidates:
        try:
            print(f"[rag] trying {model}", flush=True)
            response = client.models.generate_content(model=model, contents=prompt)
            _working_model = model
            return (response.text or "No answer returned.").strip(), hits
        except Exception as e:  # retired or overloaded model: fall through to the next one
            if not any(code in str(e) for code in RETRYABLE):
                raise
            print(f"[rag] {model} failed: {str(e)[:120]}", flush=True)
            last_error = e
    raise RuntimeError(f"All Gemini models are unavailable right now. Last error: {last_error}")


def current_model() -> str:
    return _working_model or CHAT_MODELS[0]
