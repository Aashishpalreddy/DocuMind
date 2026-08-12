from rag.ingest import load_demo_documents
from rag.vectorstore import build_vectorstore


def test_build_vectorstore_and_retrieve():
    chunks = load_demo_documents()
    store = build_vectorstore(chunks)

    results = store.similarity_search("How do I upgrade my plan?", k=3)

    assert len(results) == 3
    assert any("billing" in r.metadata["source"].lower() for r in results)
