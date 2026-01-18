import streamlit as st
import io
import time
from typing import Optional

import core as api


st.set_page_config(page_title="My Learning Buddy", layout="wide")


def load_notes():
    return api.load_state()


def create_note(title: str):
    state = api.load_state()
    note_id = api.new_id("note")
    state["notes"][note_id] = {"title": title}
    api.save_state(state)
    return note_id


def ingest_files(files, note_id: Optional[str] = None):
    raw_docs = []
    for f in files:
        data = f.read()
        text = api.extract_text(getattr(f, "name", "upload"), data)
        if text:
            meta = {"source": getattr(f, "name", "upload")}
            if note_id:
                meta["note_id"] = note_id
            raw_docs.append(api.Document(page_content=text, metadata=meta))
    if not raw_docs:
        return {"ok": False, "message": "No extractable text"}

    splitter = api.RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    splits = splitter.split_documents(raw_docs)

    vs = api.ensure_vectorstore()
    vs.add_documents(splits)
    return {"ok": True, "docs": len(raw_docs), "chunks": len(splits)}


def ask_question(question: str, note_id: Optional[str] = None):
    ctx = api.build_context(question, note_id=note_id)
    prompt = f"""
You are a friendly study assistant called My Learning Buddy.
Answer using ONLY the context. If not in context, say "I don't know".
Add short citations like [1], [2].

CONTEXT:
{ctx}

QUESTION:
{question}
"""
    llm = api.get_llm()
    out = llm.invoke(prompt).content
    if note_id:
        state = api.load_state()
        state.setdefault("chats", {})
        state["chats"].setdefault(note_id, [])
        state["chats"][note_id].append({"question": question, "answer": out, "ts": int(time.time())})
        api.save_state(state)
    return out


def generate_summary(topic: str, note_id: Optional[str] = None):
    ctx = api.build_context(topic or "main topics", note_id=note_id)
    prompt = (
        "Create a student-friendly summary with bullets + key definitions + 5 review questions.\n\n"
        f"CONTEXT:\n{ctx}"
    )
    llm = api.get_llm()
    return llm.invoke(prompt).content


def generate_flashcards(topic: str, n: int = 10, note_id: Optional[str] = None):
    ctx = api.build_context(topic or "key concepts", note_id=note_id)
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
    llm = api.get_llm()
    out = llm.invoke(prompt).content
    return api.parse_json_loose(out)


def generate_quiz(topic: str, n: int = 10, note_id: Optional[str] = None):
    ctx = api.build_context(topic or "key topics", note_id=note_id)
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
    llm = api.get_llm()
    out = llm.invoke(prompt).content
    return api.parse_json_loose(out)


def generate_mindmap(topic: str, note_id: Optional[str] = None):
    ctx = api.build_context(topic or "overview", note_id=note_id)
    prompt = f"""
Create a concise mind-map in mermaid mindmap format for the following context.
Keep it to 1 main root and up to 5 branches with 2-3 subnodes each.

CONTEXT:
{ctx}
"""
    llm = api.get_llm()
    out = llm.invoke(prompt).content
    return out


def main():
    st.markdown(
        """
        <style>
        .title {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
        .card {background: linear-gradient(135deg,#fffaf0,#fff3f8); padding:12px; border-radius:12px}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("My Learning Buddy")
    st.caption("Your friendly local RAG assistant — upload, learn, quiz, and review.")

    state = load_notes()
    notes = state.get("notes", {})
    note_map = {nid: n["title"] for nid, n in notes.items()}
    note_options = [""] + [f"{v} ({k})" for k, v in note_map.items()]

    tabs = st.tabs(["Upload", "Chat", "Quizzes", "Summarize", "Flashcards", "Mindmap"]) 

    # Upload tab
    with tabs[0]:
        st.header("Upload any files")
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded = st.file_uploader("Select files to upload", accept_multiple_files=True)
            selected_note = st.selectbox("Associate with note (optional)", options=note_options)
            if st.button("Upload & Index"):
                note_id = None
                if selected_note and "(" in selected_note:
                    note_id = selected_note.split("(")[-1][:-1]
                if not uploaded:
                    st.warning("No files selected")
                else:
                    res = ingest_files(uploaded, note_id=note_id)
                    if res.get("ok"):
                        st.success(f"Indexed {res['docs']} docs, {res['chunks']} chunks")
                    else:
                        st.error(res.get("message") or res.get("error"))
        with col2:
            st.markdown("<div class='card'><b>Tip:</b> Upload PDFs, DOCX, TXT or images (text only).</div>", unsafe_allow_html=True)

    # Chat tab
    with tabs[1]:
        st.header("Chat with your notes")
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_note = st.selectbox("Choose note", options=note_options, key="chat_note")
            note_id = None
            if selected_note and "(" in selected_note:
                note_id = selected_note.split("(")[-1][:-1]
        with col2:
            question = st.text_input("Ask a question about the selected note")
            if st.button("Ask"):
                if not question.strip():
                    st.warning("Enter a question")
                else:
                    with st.spinner("Thinking..."):
                        answer = ask_question(question.strip(), note_id=note_id)
                    st.success("Answer")
                    st.write(answer)

    # Quizzes tab
    with tabs[2]:
        st.header("Generate a Quiz")
        topic = st.text_input("Topic (optional)")
        n = st.number_input("Number of questions", min_value=1, max_value=50, value=5)
        note_choice = st.selectbox("Use note for context (optional)", options=note_options, key="quiz_note")
        if st.button("Generate Quiz"):
            note_id = None
            if note_choice and "(" in note_choice:
                note_id = note_choice.split("(")[-1][:-1]
            with st.spinner("Generating quiz..."):
                out = generate_quiz(topic, n, note_id=note_id)
            st.json(out)

    # Summarize tab
    with tabs[3]:
        st.header("Summarize notes")
        topic = st.text_input("Summary topic (optional)", key="summary_topic")
        note_choice = st.selectbox("Use note for context (optional)", options=note_options, key="summary_note")
        if st.button("Generate Summary"):
            note_id = None
            if note_choice and "(" in note_choice:
                note_id = note_choice.split("(")[-1][:-1]
            with st.spinner("Summarizing..."):
                out = generate_summary(topic, note_id=note_id)
            st.write(out)

    # Flashcards tab
    with tabs[4]:
        st.header("Create Flashcards")
        topic = st.text_input("Topic (optional)", key="flash_topic")
        n = st.number_input("Number of flashcards", min_value=1, max_value=50, value=10, key="flash_n")
        note_choice = st.selectbox("Use note for context (optional)", options=note_options, key="flash_note")
        if st.button("Generate Flashcards"):
            note_id = None
            if note_choice and "(" in note_choice:
                note_id = note_choice.split("(")[-1][:-1]
            with st.spinner("Generating flashcards..."):
                out = generate_flashcards(topic, int(n), note_id=note_id)
            st.json(out)

    # Mindmap tab
    with tabs[5]:
        st.header("Mindmap")
        topic = st.text_input("Central topic (optional)", key="mind_topic")
        note_choice = st.selectbox("Use note for context (optional)", options=note_options, key="mind_note")
        if st.button("Generate Mindmap"):
            note_id = None
            if note_choice and "(" in note_choice:
                note_id = note_choice.split("(")[-1][:-1]
            with st.spinner("Generating mindmap..."):
                out = generate_mindmap(topic, note_id=note_id)
            st.markdown("**Mermaid Mindmap**")
            st.code(out)


if __name__ == "__main__":
    main()
