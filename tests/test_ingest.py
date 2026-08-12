from rag.ingest import load_demo_documents


def test_load_demo_documents_returns_chunks():
    chunks = load_demo_documents()
    assert len(chunks) > 0


def test_demo_chunks_have_source_metadata():
    chunks = load_demo_documents()
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert chunk.metadata["source"].endswith(".txt")


def test_demo_chunks_cover_all_docs():
    chunks = load_demo_documents()
    sources = {chunk.metadata["source"] for chunk in chunks}
    assert len(sources) == 4
