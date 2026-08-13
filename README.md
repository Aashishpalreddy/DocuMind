# DocuMind — RAG Support Copilot

**The problem:** users hit an issue — an error code, a setup question, a billing
question — and the answer already exists somewhere in the product docs, but it's
not findable fast enough. So they file a support ticket and wait, and support teams
burn time answering questions the docs already cover.

**What this does:** ask a question in plain English, get an instant answer grounded
in the actual documentation, with the source cited — and a live faithfulness score
so you can tell whether the answer is actually trustworthy or hallucinated, without
a human having to double-check it. No ticket, no wait.

**Live demo: [documind-h6vwhc9aqppeamhpheg4sq.streamlit.app](https://documind-h6vwhc9aqppeamhpheg4sq.streamlit.app/)**

## How it works

1. **Ingest** — PDF/TXT docs are split into overlapping chunks.
2. **Embed** — each chunk is vectorized locally with `sentence-transformers`
   (`all-MiniLM-L6-v2`) — no API cost.
3. **Store** — vectors are indexed in **FAISS** for fast similarity search.
4. **Retrieve** — a question is embedded the same way and matched against the index
   via semantic (not keyword) search.
5. **Generate** — the retrieved chunks + question are sent to **Groq's free-tier
   Llama 3.1** API, instructed to answer only from the provided context.
6. **Evaluate** — a second LLM call judges whether the answer is faithful to the
   retrieved context (catches hallucination), and cosine similarity between the
   question and retrieved chunks scores retrieval relevance. Both are shown live
   per-answer and aggregated in an Evaluation Dashboard tab.

Two modes: a preloaded **demo knowledge base** (fictional SaaS product docs —
getting started, API reference, error codes, billing) so anyone can try it
instantly, and a **bring-your-own-document** mode for uploading your own PDF/TXT
files (nothing is persisted server-side).

## Stack

Python · Streamlit · LangChain · Groq (Llama 3.1) · sentence-transformers · FAISS ·
Docker · GitHub Actions

A `Dockerfile` is included and buildable locally (`docker build .`) to demonstrate
containerization, but the live deployment below uses Streamlit Community Cloud
directly rather than the Docker image, since Hugging Face Spaces now requires a
paid plan for Docker/Gradio SDKs (Static Spaces are the only free tier there).

Total cost: **$0**. No paid APIs, no paid hosting.

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

cp .env.example .env          # add your free Groq API key from console.groq.com
streamlit run app/app.py
```

## Running tests

```bash
pytest tests/ -v
```

## Deployment (Streamlit Community Cloud, free)

1. Go to share.streamlit.io and sign in with GitHub.
2. Click "New app", select the `Aashishpalreddy/DocuMind` repo, branch `main`,
   main file path `app/app.py`.
3. In "Advanced settings" before deploying, add a secret:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
4. Deploy. Every push to `main` auto-redeploys (Streamlit Cloud watches the repo
   directly — no GitHub Actions step needed for this part; the `CI` workflow just
   runs lint/tests on every push and PR).

## Project structure

```
app/app.py          Streamlit UI (chat + evaluation dashboard)
rag/ingest.py        document loading & chunking
rag/vectorstore.py   embeddings + FAISS index
rag/chain.py          retrieval + Groq generation
rag/evaluation.py     faithfulness (LLM-judge) + relevance (cosine similarity) scoring
data/demo_docs/       demo product documentation corpus
tests/                pytest suite (ingestion, retrieval, evaluation)
```
