"""
StudyRAG (Local) - LangChain + Ollama + FastAPI
Run locally: upload docs -> build RAG index -> chat/summary/flashcards/quiz

Prereqs:
1) Install Ollama (local): https://ollama.com/download
2) Pull models (optional; app will fallback if embeddings missing):
   ollama pull llama3.2:1b
   ollama pull nomic-embed-text
3) Create venv + install deps (see README commands below)

Start server:
  uvicorn app:app --host 127.0.0.1 --port 8000 --reload

Open:
  http://127.0.0.1:8000/       (upload page)
  http://127.0.0.1:8000/docs   (Swagger UI)

Note:
- You can change LLM_MODEL and EMBED_MODEL in this file.
"""

import io
import json
import os
import time
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

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
LLM_MODEL = "llama3.2:1b"
# Use the exact tag shown by `ollama list` for the embeddings model
# (e.g. `nomic-embed-text:latest`) to avoid "model not found" errors.
EMBED_MODEL = "nomic-embed-text:latest"

# FastAPI endpoints removed. This module now delegates to `core.py`.
# Keep a compatibility shim so other scripts can still import helpers
from core import *  # noqa: F401,F403

# Note: FastAPI server was intentionally removed in favor of the
# Streamlit UI (`streamlit_app.py`). If you need the API back, see
# version history or recreate endpoints using the helpers in `core.py`.

        if not raw_docs:
            return JSONResponse(
                {"ok": False, "message": "No extractable text. If PDF is scanned, you need OCR."},
                status_code=400,
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        splits = splitter.split_documents(raw_docs)

        vs = ensure_vectorstore()
        vs.add_documents(splits)

        return {"ok": True, "docs": len(raw_docs), "chunks": len(splits)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/chat")
def chat(req: AskReq):
    try:
        ctx = build_context(req.question, note_id=req.note_id)
        prompt = f"""
You are a study assistant.
Answer ONLY using the context. If not in context, say "I don't know".
Add citations like [1], [2].

CONTEXT:
{ctx}

QUESTION:
{req.question}
"""
        out = llm.invoke(prompt).content
        if req.note_id:
            state = load_state()
            state["chats"].setdefault(req.note_id, [])
            state["chats"][req.note_id].append(
                {"question": req.question, "answer": out, "ts": int(time.time())}
            )
            save_state(state)
        return {"answer": out}
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/summary")
def summary(req: GenReq):
    try:
        topic = req.topic or "main topics"
        ctx = build_context(topic, note_id=req.note_id)
        prompt = (
            "Create a student-friendly summary with bullets + key definitions + 5 review questions.\n\n"
            f"CONTEXT:\n{ctx}"
        )
        out = llm.invoke(prompt).content
        return {"summary": out}
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/flashcards")
def flashcards(req: GenReq):
    try:
        topic = req.topic or "key concepts"
        n = int(req.n or 12)
        ctx = build_context(topic, note_id=req.note_id)

        prompt = f"""
Return ONLY valid JSON (no markdown).
Schema:
{{
  "flashcards": [{{"front":"...","back":"..."}}]
}}
Generate exactly {n} flashcards grounded in the context.

CONTEXT:
{ctx}
"""
        out = llm.invoke(prompt).content
        return parse_json_loose(out)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/quiz")
def quiz(req: GenReq):
    try:
        topic = req.topic or "key topics"
        n = int(req.n or 10)
        ctx = build_context(topic, note_id=req.note_id)

        prompt = f"""
Return ONLY valid JSON (no markdown).
Schema:
{{
  "quiz": [
    {{
      "type": "mcq",
      "question": "...",
      "choices": ["A","B","C","D"],
      "answer_index": 0,
      "explanation": "..."
    }}
  ]
}}
Generate exactly {n} questions grounded in the context.

CONTEXT:
{ctx}
"""
        out = llm.invoke(prompt).content
        return parse_json_loose(out)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.get("/notes")
def list_notes():
    state = load_state()
    notes = [{"id": k, "title": v["title"]} for k, v in state["notes"].items()]
    return {"notes": notes}


@app.post("/notes")
def create_note(req: NoteCreateReq):
    state = load_state()
    note_id = new_id("note")
    state["notes"][note_id] = {"title": req.title}
    save_state(state)
    return {"id": note_id, "title": req.title}


@app.patch("/notes/{note_id}")
def rename_note(note_id: str, req: NoteRenameReq):
    state = load_state()
    if note_id not in state["notes"]:
        return JSONResponse({"error": "Not found"}, status_code=404)
    state["notes"][note_id]["title"] = req.title
    save_state(state)
    return {"id": note_id, "title": req.title}


@app.get("/notes/{note_id}/chats")
def get_chats(note_id: str):
    state = load_state()
    if note_id not in state["notes"]:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return {"chats": state["chats"].get(note_id, [])}
