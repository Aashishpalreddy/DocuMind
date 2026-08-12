import json
import re
from dataclasses import dataclass

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from rag.config import GROQ_API_KEY, JUDGE_MODEL
from rag.vectorstore import get_embedding_model

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. Given a CONTEXT and an ANSWER, judge whether every "
    "factual claim in the ANSWER is directly supported by the CONTEXT. Respond with "
    "ONLY a JSON object: {\"faithful\": true|false, \"score\": <0.0-1.0>, \"reasoning\": "
    "\"<one sentence>\"}. score=1.0 means fully grounded, 0.0 means entirely unsupported "
    "or hallucinated."
)


@dataclass
class EvalResult:
    faithfulness_score: float
    faithfulness_reasoning: str
    relevance_score: float


def _judge_llm() -> ChatGroq:
    return ChatGroq(api_key=GROQ_API_KEY, model=JUDGE_MODEL, temperature=0)


def score_faithfulness(answer: str, contexts: list[str]) -> tuple[float, str]:
    context_block = "\n\n".join(contexts)
    prompt = f"CONTEXT:\n{context_block}\n\nANSWER:\n{answer}"
    response = _judge_llm().invoke(
        [SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    if not match:
        return 0.0, "Judge did not return parseable output."
    try:
        parsed = json.loads(match.group(0))
        return float(parsed.get("score", 0.0)), str(parsed.get("reasoning", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0, "Judge output could not be parsed."


def score_relevance(question: str, contexts: list[str]) -> float:
    """Best-match cosine similarity between the question and retrieved chunks.

    Uses the max rather than the mean across retrieved chunks: a retriever that
    finds one strongly relevant chunk plus several weaker fillers (normal with a
    fixed top-k) should not be penalized the way a mean would penalize it.
    """
    if not contexts:
        return 0.0
    embedder = get_embedding_model()
    q_vec = np.array(embedder.embed_query(question))
    c_vecs = np.array(embedder.embed_documents(contexts))

    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    c_norms = c_vecs / (np.linalg.norm(c_vecs, axis=1, keepdims=True) + 1e-8)
    similarities = c_norms @ q_norm

    return float(np.clip(np.max(similarities), 0.0, 1.0))


def relevance_label(score: float) -> str:
    """Cosine similarity from MiniLM-style embeddings rarely exceeds ~0.5 even for
    a strong match, so the raw score should never be read as a 0-100% probability.
    Bucket it into an honest, interpretable label instead."""
    if score >= 0.35:
        return "High"
    if score >= 0.18:
        return "Medium"
    return "Low"


def evaluate(question: str, answer: str, contexts: list[str]) -> EvalResult:
    faithfulness_score, reasoning = score_faithfulness(answer, contexts)
    relevance_score = score_relevance(question, contexts)
    return EvalResult(
        faithfulness_score=faithfulness_score,
        faithfulness_reasoning=reasoning,
        relevance_score=relevance_score,
    )
