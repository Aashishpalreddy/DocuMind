# DocuMind — RAG Support Copilot

**The problem:** users hit an issue — an error code, a setup question, a billing
question — and the answer already exists somewhere in the product docs, but it's
not findable fast enough. So they file a support ticket and wait, and support teams
burn time answering questions the docs already cover.

**What this does:** ask a question in plain English, get an instant answer grounded
in the actual documentation, with the source cited — and a live faithfulness score
so you can tell whether the answer is actually trustworthy or hallucinated, without
a human having to double-check it. No ticket, no wait.

Live demo: _add your Hugging Face Space URL here after first deploy_

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

## Deployment (Hugging Face Spaces, free)

1. Create a free Space at huggingface.co/new-space with the **Docker** SDK.
2. Add three GitHub repo secrets (Settings > Secrets and variables > Actions):
   - `HF_TOKEN` — a Hugging Face access token with write access
   - `HF_USERNAME` — your Hugging Face username
   - `HF_SPACE_NAME` — the Space name you created
3. Add `GROQ_API_KEY` as a secret in the Space itself (Settings > Repository
   secrets on huggingface.co), not in GitHub.
4. Push to `main` — GitHub Actions runs tests, then force-pushes to the Space,
   which rebuilds the Docker container automatically.

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
