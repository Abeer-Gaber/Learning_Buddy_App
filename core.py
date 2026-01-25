"""Core utilities for My Learning Buddy.
Extracted from the original FastAPI app to provide a library of helpers
used by the Streamlit UI. This module contains embedding/LLM setup,
vectorstore helpers, state management and text extraction.
"""

import io
import json
import os
import time
from typing import List, Optional, Any, Dict

from pypdf import PdfReader
from docx import Document as DocxDocument

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


# ----------------------------
# Config (LOCAL)
# ----------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LLM_MODEL = "mistral"  # Changed from llama3.2:1b for better JSON output
# Use the exact tag shown by `ollama list` for the embeddings model
# (e.g. `nomic-embed-text:latest`) to avoid "model not found" errors.
EMBED_MODEL = "mxbai-embed-large"  # Better quality embeddings (free via Ollama)

PERSIST_DIR = "./chroma_db"
COLLECTION = "study_rag"
TOP_K = 5
DATA_DIR = "./data"
STATE_PATH = os.path.join(DATA_DIR, "state.json")


# LLM + embeddings (lazy init for embeddings/vectorstore)
llm: Optional[ChatOllama] = None
emb: Optional[OllamaEmbeddings] = None

vectorstore: Optional[Chroma] = None
state_cache: Optional[Dict[str, Any]] = None


# ----------------------------
# Helpers
# ----------------------------
def parse_json_loose(text: str) -> Any:
    """Best-effort JSON extraction from LLM output."""
    try:
        return json.loads(text)
    except Exception:
        pass

    # attempt to extract {...} or [...]
    for a, b in [("{", "}"), ("[", "]")]:
        i, j = text.find(a), text.rfind(b)
        if i != -1 and j != -1 and j > i:
            snippet = text[i : j + 1]
            try:
                return json.loads(snippet)
            except Exception:
                continue

    return {"raw": text, "error": "Invalid JSON from model"}


def retrieve_docs(retriever, query: str):
    """LangChain version compatibility: some retrievers use .invoke()."""
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    return retriever.invoke(query)


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join([(p.extract_text() or "") for p in reader.pages]).strip()

    if name.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs]).strip()

    # txt/md/anything else -> treat as text
    return data.decode("utf-8", errors="ignore").strip()


def ensure_embeddings() -> OllamaEmbeddings:
    """Use EMBED_MODEL if available; fallback to LLM_MODEL for embeddings."""
    global emb
    if emb is not None:
        return emb
    primary = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    try:
        primary.embed_query("healthcheck")
        emb = primary
        return emb
    except Exception:
        fallback = OllamaEmbeddings(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
        emb = fallback
        return emb


def ensure_vectorstore():
    global vectorstore
    if vectorstore is None:
        ensure_embeddings()
        vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=emb,
            persist_directory=PERSIST_DIR,
        )
    return vectorstore


def get_llm() -> ChatOllama:
    """Lazily create and return the ChatOllama instance."""
    global llm
    if llm is not None:
        return llm
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    return llm


def build_context(query: str, note_id: Optional[str] = None) -> str:
    vs = ensure_vectorstore()
    if note_id:
        retriever = vs.as_retriever(
            search_kwargs={"k": TOP_K, "filter": {"note_id": note_id}}
        )
    else:
        retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    docs = retrieve_docs(retriever, query)

    return "\n\n".join(
        [f"[{i+1}] {d.metadata.get('source','?')}\n{d.page_content}" for i, d in enumerate(docs)]
    )


def load_state() -> Dict[str, Any]:
    global state_cache
    if state_cache is not None:
        return state_cache
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(STATE_PATH):
        state_cache = {"notes": {}, "chats": {}}
        save_state(state_cache)
        return state_cache
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state_cache = json.load(f)
    state_cache.setdefault("notes", {})
    state_cache.setdefault("chats", {})
    return state_cache


def save_state(state: Dict[str, Any]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)


def new_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}"
