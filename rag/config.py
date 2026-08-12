import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVAL_K = 4

DEMO_DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "demo_docs")
