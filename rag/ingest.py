import os
from typing import BinaryIO

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, DEMO_DOCS_DIR

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _read_pdf(file: BinaryIO, source_name: str) -> list[Document]:
    reader = PdfReader(file)
    docs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": source_name, "page": page_num}))
    return docs


def _read_txt(file: BinaryIO, source_name: str) -> list[Document]:
    text = file.read().decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": source_name, "page": 1})]


def load_uploaded_files(uploaded_files) -> list[Document]:
    """uploaded_files: iterable of Streamlit UploadedFile objects."""
    raw_docs: list[Document] = []
    for f in uploaded_files:
        name = f.name
        if name.lower().endswith(".pdf"):
            raw_docs.extend(_read_pdf(f, name))
        elif name.lower().endswith(".txt"):
            raw_docs.extend(_read_txt(f, name))
    return _splitter.split_documents(raw_docs)


def load_demo_documents() -> list[Document]:
    raw_docs: list[Document] = []
    for fname in sorted(os.listdir(DEMO_DOCS_DIR)):
        path = os.path.join(DEMO_DOCS_DIR, fname)
        if fname.lower().endswith(".txt"):
            with open(path, "rb") as f:
                raw_docs.extend(_read_txt(f, fname))
        elif fname.lower().endswith(".pdf"):
            with open(path, "rb") as f:
                raw_docs.extend(_read_pdf(f, fname))
    return _splitter.split_documents(raw_docs)
