from dataclasses import dataclass
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from rag.config import GROQ_API_KEY, GROQ_MODEL, RETRIEVAL_K

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer the user's question using ONLY the "
    "provided context excerpts. If the context does not contain the answer, say "
    "you don't have enough information — do not use outside knowledge or guess."
)


@dataclass
class RagResult:
    answer: str
    contexts: list[str]
    sources: list[dict]


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file or environment.")
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)


def answer_question(vectorstore: FAISS, question: str) -> RagResult:
    retrieved = vectorstore.similarity_search(question, k=RETRIEVAL_K)
    contexts = [doc.page_content for doc in retrieved]
    sources = [
        {"source": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page", "-")}
        for doc in retrieved
    ]

    context_block = "\n\n---\n\n".join(
        f"[Excerpt {i + 1} — {s['source']}, page {s['page']}]\n{c}"
        for i, (c, s) in enumerate(zip(contexts, sources))
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)])

    return RagResult(answer=response.content, contexts=contexts, sources=sources)
