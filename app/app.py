import os
import sys
import time

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chain import answer_question
from rag.config import GROQ_API_KEY
from rag.evaluation import evaluate, relevance_label
from rag.ingest import load_demo_documents, load_uploaded_files
from rag.vectorstore import build_vectorstore

st.set_page_config(page_title="DocuMind — RAG Support Copilot", page_icon="📄", layout="wide")

if "query_log" not in st.session_state:
    st.session_state.query_log = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "corpus_label" not in st.session_state:
    st.session_state.corpus_label = None


@st.cache_resource(show_spinner=False)
def _demo_vectorstore():
    chunks = load_demo_documents()
    return build_vectorstore(chunks)


st.title("📄 DocuMind — RAG Support Copilot")
st.caption(
    "Ask a question and get an instant, cited answer grounded in the documentation — "
    "no support ticket, no waiting. Every answer is scored live for faithfulness and "
    "retrieval relevance so you can see whether it's actually trustworthy."
)

with st.sidebar:
    st.header("Knowledge base")
    mode = st.radio("Source", ["Demo docs (CloudSync product docs)", "Upload your own"])

    if not GROQ_API_KEY:
        st.warning("GROQ_API_KEY is not set. Add it to a .env file to enable answers.")

    if mode.startswith("Demo"):
        if st.session_state.corpus_label != "demo":
            with st.spinner("Indexing demo documents..."):
                st.session_state.vectorstore = _demo_vectorstore()
                st.session_state.corpus_label = "demo"
        st.success("Loaded: CloudSync product docs (getting started, API reference, error codes, billing)")
    else:
        uploaded = st.file_uploader(
            "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
        )
        if uploaded and st.button("Index documents"):
            with st.spinner("Chunking and embedding your documents..."):
                chunks = load_uploaded_files(uploaded)
                st.session_state.vectorstore = build_vectorstore(chunks)
                st.session_state.corpus_label = "custom"
            st.success(f"Indexed {len(chunks)} chunks from {len(uploaded)} file(s).")

tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation Dashboard"])

with tab_chat:
    if st.session_state.vectorstore is None:
        st.info("Select or upload a knowledge base in the sidebar to get started.")
    else:
        question = st.text_input("Ask a question about the documentation:")
        if question:
            with st.spinner("Retrieving context and generating answer..."):
                t0 = time.time()
                result = answer_question(st.session_state.vectorstore, question)
                latency = time.time() - t0

            with st.spinner("Scoring faithfulness and relevance..."):
                eval_result = evaluate(question, result.answer, result.contexts)

            st.markdown("### Answer")
            st.write(result.answer)

            col1, col2, col3 = st.columns(3)
            col1.metric("Faithfulness", f"{eval_result.faithfulness_score:.2f}")
            col2.metric(
                "Retrieval relevance",
                relevance_label(eval_result.relevance_score),
                f"similarity {eval_result.relevance_score:.2f}",
            )
            col3.metric("Latency", f"{latency:.1f}s")

            if eval_result.faithfulness_score < 0.6:
                st.error(f"⚠️ Low faithfulness — {eval_result.faithfulness_reasoning}")
            else:
                st.caption(f"Judge reasoning: {eval_result.faithfulness_reasoning}")

            with st.expander("Sources used"):
                for i, (ctx, src) in enumerate(zip(result.contexts, result.sources), start=1):
                    st.markdown(f"**[{i}] {src['source']} — page {src['page']}**")
                    st.text(ctx[:500] + ("..." if len(ctx) > 500 else ""))

            st.session_state.query_log.append(
                {
                    "question": question,
                    "answer": result.answer,
                    "faithfulness": eval_result.faithfulness_score,
                    "relevance": eval_result.relevance_score,
                    "latency": latency,
                }
            )

with tab_eval:
    st.markdown("### Session evaluation log")
    st.caption(
        "Faithfulness is LLM-judged (1.0 = fully grounded, 0.0 = hallucinated). "
        "Relevance is best-match embedding cosine similarity between the question and "
        "retrieved chunks — these embeddings rarely exceed ~0.5 even for a strong "
        "match, so it's bucketed into Low/Medium/High rather than read as a percentage."
    )
    if not st.session_state.query_log:
        st.info("Ask a question in the Chat tab to populate this dashboard.")
    else:
        st.dataframe(
            [
                {
                    "Question": q["question"],
                    "Faithfulness": round(q["faithfulness"], 2),
                    "Relevance": relevance_label(q["relevance"]),
                    "Relevance (raw)": round(q["relevance"], 2),
                    "Latency (s)": round(q["latency"], 1),
                }
                for q in st.session_state.query_log
            ],
            use_container_width=True,
        )

        col1, col2, col3 = st.columns(3)
        n = len(st.session_state.query_log)
        col1.metric("Avg. faithfulness", f"{sum(q['faithfulness'] for q in st.session_state.query_log) / n:.2f}")
        col2.metric("Avg. relevance (similarity)", f"{sum(q['relevance'] for q in st.session_state.query_log) / n:.2f}")
        col3.metric("Avg. latency", f"{sum(q['latency'] for q in st.session_state.query_log) / n:.1f}s")

with st.sidebar:
    st.divider()
    st.metric("Queries this session", len(st.session_state.query_log))
    if st.session_state.query_log:
        avg_faith = sum(q["faithfulness"] for q in st.session_state.query_log) / len(
            st.session_state.query_log
        )
        st.metric("Avg. faithfulness", f"{avg_faith:.2f}")
