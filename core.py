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
import re
import math

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


def calculator_tool(expression: str) -> str:
    """
    Safe calculator tool for basic math operations.
    Supports: +, -, *, /, ^, sqrt, sin, cos, tan, log, pi, e
    """
    try:
        # Clean and validate the expression
        allowed_chars = set('0123456789+-*/().^ sqrtincologepi')
        expr = expression.lower().replace(' ', '')
        
        if not all(c in allowed_chars for c in expr.replace('sqrt', '').replace('sin', '').replace('cos', '').replace('tan', '').replace('log', '').replace('pi', '').replace('e', '')):
            return "Error: Invalid characters in expression"
        
        # Replace math symbols
        expr = expr.replace('^', '**')
        expr = expr.replace('sqrt', 'math.sqrt')
        expr = expr.replace('sin', 'math.sin')
        expr = expr.replace('cos', 'math.cos')
        expr = expr.replace('tan', 'math.tan')
        expr = expr.replace('log', 'math.log10')
        expr = expr.replace('pi', str(math.pi))
        expr = expr.replace('e', str(math.e))
        
        # Evaluate safely
        result = eval(expr, {"__builtins__": {}, "math": math})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def ask_with_tools(question: str, note_id: str = None) -> str:
    """
    Enhanced Q&A that can use tools when needed.
    The LLM decides whether to use a tool based on the question.
    """
    llm = get_llm()
    
    # First, ask the LLM if it needs a tool
    tool_check_prompt = f"""You are a study assistant with access to tools.

Available tools:
1. CALCULATOR - For math calculations. Use format: [CALC: expression]
   Example: [CALC: 25 * 4 + 10] or [CALC: sqrt(144)]

Question: {question}

If this question requires a calculation, respond ONLY with the calculator call like [CALC: expression].
If no calculation is needed, respond with: NO_TOOL_NEEDED"""

    tool_response = llm.invoke(tool_check_prompt).content.strip()
    
    # Check if LLM wants to use calculator
    calc_match = re.search(r'\[CALC:\s*([^\]]+)\]', tool_response)
    
    tool_result = None
    if calc_match:
        expression = calc_match.group(1)
        tool_result = calculator_tool(expression)
    
    # Now generate final answer
    ctx = build_context(question, note_id=note_id)
    
    if tool_result:
        final_prompt = f"""You are "My Learning Buddy" — a friendly study helper.

You used the calculator tool and got: {tool_result}

Context from notes:
{ctx}

Question: {question}

Provide a helpful answer incorporating the calculation result. Explain the math if relevant."""
    else:
        final_prompt = f"""You are "My Learning Buddy" — a friendly study helper.

Context from notes:
{ctx}

Question: {question}

Answer based on the context above."""

    return llm.invoke(final_prompt).content


def researcher_agent(question: str, note_id: str = None) -> str:
    """
    Agent 1: Researcher - Finds and extracts relevant information.
    Returns raw facts without explanation.
    """
    llm = get_llm()
    ctx = build_context(question, note_id=note_id)
    
    if not ctx or not ctx.strip():
        return "No relevant information found in the uploaded documents."
    
    prompt = f"""You are a RESEARCHER agent. Your job is to:
1. Find relevant facts from the provided context
2. Extract key information that answers the question
3. List facts as bullet points - be precise and factual
4. Include source references [1], [2], etc.

DO NOT explain or teach. Just extract the raw facts.

CONTEXT:
{ctx}

QUESTION: {question}

EXTRACTED FACTS:"""
    
    return llm.invoke(prompt).content


def teacher_agent(question: str, research_facts: str) -> str:
    """
    Agent 2: Teacher - Takes research and explains it clearly.
    Makes content student-friendly and engaging.
    """
    llm = get_llm()
    
    prompt = f"""You are a TEACHER agent named "My Learning Buddy". Your job is to:
1. Take the research facts provided below
2. Explain them in a friendly, easy-to-understand way
3. Add helpful examples or analogies if useful
4. Use encouraging language suitable for students
5. Structure the answer clearly

DO NOT add new facts - only explain what the Researcher found.
If the research says "no information found", kindly tell the student to upload relevant materials.

RESEARCH FACTS:
{research_facts}

STUDENT'S QUESTION: {question}

YOUR FRIENDLY EXPLANATION:"""
    
    return llm.invoke(prompt).content


def ask_with_agents(question: str, note_id: str = None) -> dict:
    """
    Multi-agent Q&A: Researcher finds info, Teacher explains it.
    Returns both intermediate and final results for transparency.
    """
    # Agent 1: Research
    research_output = researcher_agent(question, note_id)
    
    # Agent 2: Teach
    teacher_output = teacher_agent(question, research_output)
    
    return {
        "researcher_output": research_output,
        "teacher_output": teacher_output,
        "final_answer": teacher_output
    }