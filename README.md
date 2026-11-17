# PS36 – Research Paper Q&A (LLM Mini Project)

**Problem statement:** Students read many papers for a literature review. This app lets you upload a research paper (PDF) and ask questions about its method, dataset and results, with answers grounded in the paper and cited by page number.

## Features
- Upload any text-based research paper PDF
- RAG pipeline: PDF → chunks → Gemini embeddings → FAISS → Gemini 2.5 Flash
- Four preset buttons: **Method, Dataset, Results, Limitations**, each sending a fixed question
- Free-text chat for anything else
- Page-number citations, plus a "Sources used" panel that shows the retrieved chunks
- Replies "Not found in the paper." instead of guessing

## Tech stack
Python · Streamlit · Google Gemini API (`gemini-2.5-flash`, `gemini-embedding-001`) · FAISS · pypdf

## How to run
```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your Gemini API key into .env
streamlit run app.py
```
Get a free API key at https://aistudio.google.com/apikey. You can also paste the key in the app's sidebar.

## Project structure
```
app.py            Streamlit UI (upload, preset buttons, chat, sources)
rag.py            PDF reading, chunking, embeddings, FAISS search, prompt
requirements.txt
.env.example      Template for the API key (the real .env is git-ignored)
```

## Screenshots
_Add 1–2 screenshots here (for example `screenshots/method.png`)._
