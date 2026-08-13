import os
import sys
import time

import altair as alt
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chain import answer_question
from rag.config import GROQ_API_KEY
from rag.evaluation import evaluate, relevance_label
from rag.ingest import load_demo_documents, load_uploaded_files
from rag.vectorstore import build_vectorstore

st.set_page_config(page_title="DocuMind — RAG Support Copilot", page_icon="📄", layout="wide")

SUGGESTED_QUESTIONS = [
    "How do I upgrade my plan and what does Pro cost?",
    "I'm getting a 409 error, what does that mean?",
    "How do I install the CloudSync CLI?",
    "What happens if I exceed my storage limit?",
]

CHART_BLUE = "#3987e5"
CHART_ORANGE = "#d95926"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_log" not in st.session_state:
    st.session_state.query_log = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "corpus_label" not in st.session_state:
    st.session_state.corpus_label = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


@st.cache_resource(show_spinner=False)
def _demo_vectorstore():
    chunks = load_demo_documents()
    return build_vectorstore(chunks)


def ask(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})

    t0 = time.time()
    result = answer_question(st.session_state.vectorstore, question)
    latency = time.time() - t0
    eval_result = evaluate(question, result.answer, result.contexts)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "faithfulness": eval_result.faithfulness_score,
            "faithfulness_reasoning": eval_result.faithfulness_reasoning,
            "relevance": eval_result.relevance_score,
            "latency": latency,
            "sources": result.sources,
            "contexts": result.contexts,
        }
    )
    st.session_state.query_log.append(
        {
            "question": question,
            "answer": result.answer,
            "faithfulness": eval_result.faithfulness_score,
            "relevance": eval_result.relevance_score,
            "latency": latency,
        }
    )


st.title("📄 DocuMind — RAG Support Copilot")
st.caption(
    "Ask a question and get an instant, cited answer grounded in the documentation — "
    "no support ticket, no waiting. Every answer is scored live for faithfulness and "
    "retrieval relevance so you can see whether it's actually trustworthy."
)

step1, step2, step3, step4 = st.columns(4)
with step1, st.container(border=True):
    st.markdown("**1 · Retrieve**")
    st.caption("Your question is embedded and matched against the docs via FAISS semantic search.")
with step2, st.container(border=True):
    st.markdown("**2 · Generate**")
    st.caption("Retrieved chunks + question go to Llama 3.1 (Groq), told to answer only from context.")
with step3, st.container(border=True):
    st.markdown("**3 · Verify**")
    st.caption("A second LLM call judges whether the answer is actually grounded — catches hallucination.")
with step4, st.container(border=True):
    st.markdown("**4 · Score**")
    st.caption("Faithfulness + retrieval relevance are logged live to the Evaluation Dashboard.")

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
            st.session_state.messages = []
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
                st.session_state.messages = []
            st.success(f"Indexed {len(chunks)} chunks from {len(uploaded)} file(s).")

tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation Dashboard"])

with tab_chat:
    if st.session_state.vectorstore is None:
        st.info("Select or upload a knowledge base in the sidebar to get started.")
    else:
        if not st.session_state.messages:
            st.markdown("**Try asking:**")
            chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
            for col, suggested_q in zip(chip_cols, SUGGESTED_QUESTIONS):
                if col.button(suggested_q, use_container_width=True):
                    st.session_state.pending_question = suggested_q

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant":
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Faithfulness", f"{msg['faithfulness']:.2f}")
                    c2.metric(
                        "Retrieval relevance",
                        relevance_label(msg["relevance"]),
                        f"similarity {msg['relevance']:.2f}",
                    )
                    c3.metric("Latency", f"{msg['latency']:.1f}s")

                    if msg["faithfulness"] < 0.6:
                        st.error(f"⚠️ Low faithfulness — {msg['faithfulness_reasoning']}")
                    else:
                        st.caption(f"Judge reasoning: {msg['faithfulness_reasoning']}")

                    with st.expander("Sources used"):
                        for i, (ctx, src) in enumerate(zip(msg["contexts"], msg["sources"]), start=1):
                            st.markdown(f"**[{i}] {src['source']} — page {src['page']}**")
                            st.text(ctx[:500] + ("..." if len(ctx) > 500 else ""))

        typed_question = st.chat_input("Ask a question about the documentation...")
        question_to_ask = st.session_state.pending_question or typed_question

        if question_to_ask:
            st.session_state.pending_question = None
            with st.spinner("Retrieving context, generating, and scoring the answer..."):
                ask(question_to_ask)
            st.rerun()

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
        df = pd.DataFrame(st.session_state.query_log)
        df.insert(0, "Query #", range(1, len(df) + 1))

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg. faithfulness", f"{df['faithfulness'].mean():.2f}")
        col2.metric("Avg. relevance (similarity)", f"{df['relevance'].mean():.2f}")
        col3.metric("Avg. latency", f"{df['latency'].mean():.1f}s")

        st.markdown("#### Faithfulness & relevance over the session")
        trend_df = df.melt(
            id_vars=["Query #"],
            value_vars=["faithfulness", "relevance"],
            var_name="Metric",
            value_name="Score",
        )
        trend_df["Metric"] = trend_df["Metric"].map(
            {"faithfulness": "Faithfulness", "relevance": "Relevance"}
        )
        trend_chart = (
            alt.Chart(trend_df)
            .mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2)
            .encode(
                x=alt.X("Query #:O", axis=alt.Axis(title="Query #")),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(title="Score")),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(domain=["Faithfulness", "Relevance"], range=[CHART_BLUE, CHART_ORANGE]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["Query #", "Metric", alt.Tooltip("Score:Q", format=".2f")],
            )
            .properties(height=260)
        )
        st.altair_chart(trend_chart, use_container_width=True)

        st.markdown("#### Latency per query")
        latency_chart = (
            alt.Chart(df)
            .mark_bar(color=CHART_BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Query #:O", axis=alt.Axis(title="Query #")),
                y=alt.Y("latency:Q", axis=alt.Axis(title="Latency (s)")),
                tooltip=["Query #", alt.Tooltip("latency:Q", title="Latency (s)", format=".1f")],
            )
            .properties(height=200)
        )
        st.altair_chart(latency_chart, use_container_width=True)

        st.markdown("#### Full log")
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

with st.sidebar:
    st.divider()
    st.metric("Queries this session", len(st.session_state.query_log))
    if st.session_state.query_log:
        avg_faith = sum(q["faithfulness"] for q in st.session_state.query_log) / len(
            st.session_state.query_log
        )
        st.metric("Avg. faithfulness", f"{avg_faith:.2f}")
