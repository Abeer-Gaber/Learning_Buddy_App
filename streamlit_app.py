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
    import shutil
    import os
    
    # Clear old data first (fresh start with each upload)
    chroma_path = "./chroma_db"
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
    # Reset the vectorstore reference
    api.vectorstore = None
    
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
    
    # Check if we have meaningful context
    has_context = ctx and ctx.strip() and ctx.strip() not in ["", "No relevant context found.", "None"]
    
    if has_context:
        prompt = f"""You are "My Learning Buddy" — a friendly study helper.

YOUR JOB: Answer the user's question using ONLY the context provided below.

RULES:
1. If the answer IS in the context → Answer clearly and friendly, cite with [1], [2] where helpful
2. If the answer is NOT in the context → Say "I couldn't find information about [topic] in your notes."
3. NEVER add information from outside the context
4. NEVER guess or make things up
5. Keep answers clear and helpful

CONTEXT FROM USER'S NOTES:
{ctx}

QUESTION: {question}

Answer based ONLY on the context above:"""
    else:
        prompt = f"""You are "My Learning Buddy" — a friendly study helper.

The user asked: "{question}"

You searched their notes but found no relevant information about this topic.

Respond with a short, friendly message (2-3 sentences):
- Let them know you couldn't find this in their notes
- Suggest uploading materials about this topic
- Do NOT explain or define the term — just say it's not in their notes"""

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
    
    topic_text = f'about "{topic}"' if topic.strip() else "from the materials"
    
    prompt = f"""You are "My Learning Buddy" — a friendly study helper creating summaries from the user's own notes.

RULES:
- Use ONLY information from the CONTEXT below — don't add outside knowledge
- Keep it student-friendly and easy to understand
- Be encouraging and supportive in tone

Create a helpful study summary {topic_text} with:
📌 **Key Points** (clear bullet points of the main ideas)
📖 **Important Definitions** (key terms explained simply)
❓ **Review Questions** (5 questions to test understanding)

CONTEXT FROM USER'S NOTES:
{ctx}

Make it clear, organized, and helpful for studying!"""
    
    llm = api.get_llm()
    return llm.invoke(prompt).content


def generate_flashcards(topic: str, n: int = 10, note_id: Optional[str] = None):
    import re
    import json
    
    ctx = api.build_context(topic or "key concepts", note_id=note_id)
    
    # Check if we have context
    if not ctx or not ctx.strip() or ctx.strip() in ["No relevant context found.", "None", ""]:
        return {"flashcards": [], "error": "No documents uploaded. Please upload study materials first."}
    
    # Very simple prompt for small models
    prompt = f"""Read this text and create {n} flashcards.

TEXT:
{ctx}

Create flashcards as JSON. Example:
{{"flashcards": [{{"front": "What is DNA?", "back": "DNA is the molecule that carries genetic information."}}]}}

Your {n} flashcards as JSON:"""

    llm = api.get_llm()
    raw_output = llm.invoke(prompt).content
    
    # Clean up the response
    out = raw_output.strip()
    
    # Remove markdown code blocks
    out = re.sub(r'```json\s*\n?', '', out)
    out = re.sub(r'```\s*\n?', '', out)
    out = out.strip()
    
    # Fix common JSON issues from small models
    out = re.sub(r',\s*([}\]])', r'\1', out)  # Remove trailing commas
    
    # Try to fix single quotes to double quotes
    # This is tricky - we need to be careful about apostrophes in text
    if "'" in out and '"' not in out:
        # Likely using single quotes for JSON
        out = out.replace("'", '"')
    
    # Try to parse JSON with multiple strategies
    result = None
    
    # Strategy 1: Direct parse
    try:
        result = json.loads(out)
    except:
        pass
    
    # Strategy 2: Find JSON object containing "flashcards"
    if result is None:
        try:
            # Find the start of the JSON object
            start = out.find('{')
            if start >= 0:
                depth = 0
                end = start
                for i, char in enumerate(out[start:], start):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                json_str = out[start:end]
                result = json.loads(json_str)
        except:
            pass
    
    # Strategy 3: Find array and wrap it
    if result is None:
        try:
            arr_start = out.find('[')
            if arr_start >= 0:
                depth = 0
                end = arr_start
                for i, char in enumerate(out[arr_start:], arr_start):
                    if char == '[':
                        depth += 1
                    elif char == ']':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                arr_str = out[arr_start:end]
                arr = json.loads(arr_str)
                if isinstance(arr, list):
                    result = {"flashcards": arr}
        except:
            pass
    
    # Strategy 4: Try to extract front/back pairs with regex
    if result is None:
        try:
            # Look for patterns like "front": "...", "back": "..."
            pattern = r'"front"\s*:\s*"([^"]+)"\s*,\s*"back"\s*:\s*"([^"]+)"'
            matches = re.findall(pattern, out)
            if matches:
                result = {"flashcards": [{"front": m[0], "back": m[1]} for m in matches]}
        except:
            pass
    
    # Strategy 5: Fallback to api parser
    if result is None:
        result = api.parse_json_loose(out)
    
    # Always include raw for debugging
    if isinstance(result, dict):
        result["_raw"] = raw_output[:1500]
    else:
        result = {"flashcards": [], "_raw": raw_output[:1500], "error": "Could not parse response"}
    
    # Validate and clean flashcards
    if isinstance(result, dict) and "flashcards" in result and isinstance(result["flashcards"], list):
        clean_cards = []
        for card in result["flashcards"]:
            if not isinstance(card, dict):
                continue
            
            # Get front and back, handle different key names
            front = card.get("front") or card.get("question") or card.get("q") or ""
            back = card.get("back") or card.get("answer") or card.get("a") or ""
            
            front = str(front).strip()
            back = str(back).strip()
            
            # Skip empty or placeholder cards
            if not front or not back:
                continue
            if front in ["...", "[term]", "What is [term]?"]:
                continue
            if back in ["...", "[Definition - 2-3 sentences]", "[definition]"]:
                continue
            if len(front) < 5 or len(back) < 5:
                continue
                
            # Ensure front is a question (add ? if missing)
            if not front.endswith("?") and not front.endswith(":"):
                if front.lower().startswith(("what", "who", "when", "where", "why", "how", "which", "explain", "describe", "define")):
                    front = front + "?"
            
            clean_cards.append({"front": front, "back": back})
        
        result["flashcards"] = clean_cards
        
        if not clean_cards and "error" not in result:
            result["error"] = "No valid flashcards could be extracted from the model response."
    
    elif "error" not in result:
        result["flashcards"] = []
        result["error"] = "Could not parse flashcards from model response"
    
    return result


def generate_quiz(topic: str, n: int = 10, note_id: Optional[str] = None):
    import re
    import json
    
    ctx = api.build_context(topic or "key topics", note_id=note_id)
    
    # Check if we have context
    if not ctx or not ctx.strip() or ctx.strip() in ["No relevant context found.", "None", ""]:
        return {"quiz": [], "error": "No documents uploaded. Please upload study materials first."}
    
    topic_text = f'about "{topic}"' if topic and topic.strip() else "from the materials"
    
    # Clearer prompt with real examples - not placeholder letters
    prompt = f"""Read the text and create {n} multiple choice questions.

TEXT:
{ctx}

IMPORTANT: Each choice must be a real answer, NOT just a letter!

Return JSON like this example:
{{"quiz": [
  {{"question": "What is the main function of the heart?", "choices": ["To pump blood throughout the body", "To digest food", "To filter air", "To produce hormones"], "answer_index": 0, "explanation": "The heart pumps blood to all parts of the body."}}
]}}

Create {n} questions with 4 real answer choices each. JSON:"""

    llm = api.get_llm()
    raw_output = llm.invoke(prompt).content
    
    # Clean up the response
    out = raw_output.strip()
    
    # Remove markdown code blocks
    out = re.sub(r'```json\s*\n?', '', out)
    out = re.sub(r'```\s*\n?', '', out)
    out = out.strip()
    
    # Fix common JSON issues from small models
    # Replace single quotes with double quotes (but not inside strings)
    out = re.sub(r"'(\s*:\s*)", r'"\1', out)  # 'key': -> "key":
    out = re.sub(r":\s*'([^']*)'", r': "\1"', out)  # : 'value' -> : "value"
    
    # Remove trailing commas before } or ]
    out = re.sub(r',\s*([}\]])', r'\1', out)
    
    # Try multiple parsing strategies
    result = None
    
    # Strategy 1: Direct parse
    try:
        result = json.loads(out)
    except:
        pass
    
    # Strategy 2: Find the outermost { } containing "quiz"
    if result is None:
        try:
            # Find where "quiz" appears and work backwards/forwards
            quiz_pos = out.find('"quiz"')
            if quiz_pos == -1:
                quiz_pos = out.find("'quiz'")
            
            if quiz_pos >= 0:
                # Find the { before quiz
                start = out.rfind('{', 0, quiz_pos)
                if start >= 0:
                    # Find matching }
                    depth = 0
                    end = start
                    in_string = False
                    escape = False
                    for i, char in enumerate(out[start:], start):
                        if escape:
                            escape = False
                            continue
                        if char == '\\':
                            escape = True
                            continue
                        if char == '"' and not escape:
                            in_string = not in_string
                            continue
                        if in_string:
                            continue
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    
                    json_str = out[start:end]
                    result = json.loads(json_str)
        except:
            pass
    
    # Strategy 3: Try to find any valid JSON array for quiz
    if result is None:
        try:
            # Look for array pattern
            arr_start = out.find('[')
            if arr_start >= 0:
                depth = 0
                end = arr_start
                for i, char in enumerate(out[arr_start:], arr_start):
                    if char == '[':
                        depth += 1
                    elif char == ']':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                arr_str = out[arr_start:end]
                arr = json.loads(arr_str)
                if isinstance(arr, list) and len(arr) > 0:
                    result = {"quiz": arr}
        except:
            pass
    
    # Strategy 4: Fallback to api parser
    if result is None:
        result = api.parse_json_loose(out)
    
    # Always include raw output for debugging
    if isinstance(result, dict):
        result["_raw"] = raw_output[:1000]
    
    # Validate and fix the result
    if isinstance(result, dict) and "quiz" in result and isinstance(result["quiz"], list):
        valid_questions = []
        for q in result["quiz"]:
            if not isinstance(q, dict):
                continue
            
            # Check if choices are valid (not just letters or too short)
            if "choices" in q and isinstance(q["choices"], list):
                choices = q["choices"]
                # Filter out invalid choices (single letters, empty, too short)
                invalid_patterns = ["A", "B", "C", "D", "a", "b", "c", "d", "...", ""]
                has_invalid_choices = all(
                    str(c).strip() in invalid_patterns or len(str(c).strip()) < 3 
                    for c in choices
                )
                if has_invalid_choices:
                    # Skip this question - choices are placeholders
                    continue
                
                # Clean up choices
                clean_choices = []
                for c in choices:
                    choice_str = str(c).strip()
                    # Remove leading "A.", "B.", etc. if model added them
                    if len(choice_str) > 2 and choice_str[0] in "ABCDabcd" and choice_str[1] in ".):":
                        choice_str = choice_str[2:].strip()
                    if choice_str and len(choice_str) >= 3:
                        clean_choices.append(choice_str)
                
                if len(clean_choices) < 2:
                    continue  # Not enough valid choices
                    
                # Pad to 4 choices if needed
                while len(clean_choices) < 4:
                    clean_choices.append(f"Option {chr(65 + len(clean_choices))}")
                q["choices"] = clean_choices[:4]
            else:
                continue  # No choices, skip question
            
            # Ensure answer_index is valid (0-3)
            if "answer_index" in q:
                idx = q["answer_index"]
                if not isinstance(idx, int):
                    try:
                        idx = int(idx)
                        q["answer_index"] = idx
                    except:
                        q["answer_index"] = 0
                if idx < 0 or idx > 3:
                    q["answer_index"] = 0
            else:
                q["answer_index"] = 0
            
            # Ensure question exists and is meaningful
            if "question" not in q or not q["question"] or len(str(q["question"]).strip()) < 10:
                continue  # Skip questions that are too short
            
            # This question passed all validation
            valid_questions.append(q)
        
        # Use only valid questions
        result["quiz"] = valid_questions
        
        # If no valid questions, add error message
        if not valid_questions:
            result["error"] = "Could not generate valid quiz questions. The model may have returned placeholder content. Try again."
        
    elif isinstance(result, dict) and "error" in result:
        result["quiz"] = []
        result["_raw"] = raw_output[:1000]
    else:
        result = {"quiz": [], "_raw": raw_output[:1000], "error": "Could not parse quiz from model response"}
    
    return result


def generate_mindmap(topic: str, note_id: Optional[str] = None):
    ctx = api.build_context(topic or "overview", note_id=note_id)
    
    # Check if we have context
    if not ctx or not ctx.strip() or ctx.strip() in ["No relevant context found.", "None", ""]:
        return '{"title": "No Content", "branches": [{"name": "Upload documents first", "items": ["Go to Upload tab", "Add your study materials"]}]}'
    
    topic_text = f'about "{topic}"' if topic and topic.strip() else ""
    
    prompt = f"""Create ONE mindmap {topic_text} from the context below.

CRITICAL RULES:
1. Use ONLY facts from the CONTEXT - never make up content
2. Return exactly ONE JSON object
3. If context is about cooking, make a cooking mindmap. If about history, make a history mindmap. Match the actual content!

Format (return ONLY this, no extra text):
{{"title": "Topic from context", "branches": [{{"name": "Theme 1", "items": ["fact 1", "fact 2"]}}, {{"name": "Theme 2", "items": ["fact 1", "fact 2"]}}]}}

CONTEXT:
{ctx}

Return ONE JSON object only:"""
    
    llm = api.get_llm()
    out = llm.invoke(prompt).content.strip()
    
    # Clean up - extract only the first valid JSON object
    import json
    import re
    
    # Remove markdown code blocks
    out = re.sub(r'```json\s*', '', out)
    out = re.sub(r'```\s*', '', out)
    
    # Try to find and parse just the first JSON object
    try:
        # Find first { and matching }
        start = out.find('{')
        if start >= 0:
            depth = 0
            end = start
            for i, char in enumerate(out[start:], start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            
            json_str = out[start:end]
            # Validate it's proper JSON
            parsed = json.loads(json_str)
            return json.dumps(parsed)  # Return clean JSON
    except:
        pass
    
    return out


def render_interactive_quiz(quiz_data):
    """Render quiz as interactive questions with click-to-reveal answers."""
    
    # Get quiz list from data
    questions = []
    if isinstance(quiz_data, dict):
        questions = quiz_data.get("quiz", [])
    elif isinstance(quiz_data, list):
        questions = quiz_data
    
    if not questions:
        # Show error message if available
        if isinstance(quiz_data, dict):
            if "error" in quiz_data:
                st.error(f"⚠️ {quiz_data['error']}")
            # Show raw output for debugging (check both keys)
            raw = quiz_data.get("_raw") or quiz_data.get("raw")
            if raw:
                with st.expander("🔍 View raw model response (for debugging)"):
                    st.code(raw)
                st.info("💡 Tip: The model may be struggling with JSON format. Try generating again or use a larger model.")
        st.warning("No quiz questions generated. Make sure you have uploaded documents first, then try again.")
        return
    
    # Initialize session state for revealed answers
    if "revealed_answers" not in st.session_state:
        st.session_state.revealed_answers = set()
    
    # Initialize session state for user selections
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .quiz-question-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 20px 25px;
            margin: 20px 0 15px 0;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        .quiz-question-number {
            font-size: 12px;
            opacity: 0.85;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .quiz-question-text {
            font-size: 18px;
            font-weight: 600;
            line-height: 1.5;
        }
        .answer-correct {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            border-radius: 12px;
            padding: 15px 20px;
            margin: 15px 0;
            box-shadow: 0 5px 20px rgba(17, 153, 142, 0.3);
        }
        .answer-wrong {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            color: white;
            border-radius: 12px;
            padding: 15px 20px;
            margin: 15px 0;
            box-shadow: 0 5px 20px rgba(235, 51, 73, 0.3);
        }
        .answer-neutral {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border-radius: 12px;
            padding: 15px 20px;
            margin: 15px 0;
            box-shadow: 0 5px 20px rgba(79, 172, 254, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 Interactive Quiz")
    st.caption("Select your answer for each question, then click 'Show Answer' to check!")
    
    for q_idx, question in enumerate(questions):
        q_text = question.get("question", "Question")
        choices = question.get("choices", [])
        
        # Validate and fix correct_idx
        correct_idx = question.get("answer_index", 0)
        if not isinstance(correct_idx, int):
            try:
                correct_idx = int(correct_idx)
            except:
                correct_idx = 0
        # Ensure it's within valid range
        if correct_idx < 0 or correct_idx >= len(choices):
            correct_idx = 0
        
        explanation = question.get("explanation", "See your notes for more details.")
        
        is_revealed = q_idx in st.session_state.revealed_answers
        user_answer = st.session_state.user_answers.get(q_idx)
        
        # Question card with better styling
        st.markdown(f"""
            <div class="quiz-question-card">
                <div class="quiz-question-number">Question {q_idx + 1} of {len(questions)}</div>
                <div class="quiz-question-text">{q_text}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display choices as styled radio buttons
        if choices:
            # Create closure to capture choice_labels correctly
            choice_labels = [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
            
            selected = st.radio(
                f"Select answer for Q{q_idx + 1}:",
                options=range(len(choices)),
                format_func=lambda x, labels=choice_labels: labels[x],
                key=f"quiz_q_{q_idx}",
                index=user_answer if user_answer is not None else None,
                label_visibility="collapsed"
            )
            st.session_state.user_answers[q_idx] = selected
        
        # Show Answer button - styled and prominent
        btn_label = "🙈 Hide Answer" if is_revealed else "👁️ Show Answer"
        if st.button(btn_label, key=f"reveal_{q_idx}", type="secondary" if is_revealed else "primary"):
            if is_revealed:
                st.session_state.revealed_answers.discard(q_idx)
            else:
                st.session_state.revealed_answers.add(q_idx)
            st.rerun()
        
        # Show answer if revealed
        if is_revealed:
            user_ans = st.session_state.user_answers.get(q_idx)
            correct_letter = chr(65 + correct_idx)
            correct_text = choices[correct_idx] if correct_idx < len(choices) else ""
            
            if user_ans is not None:
                is_correct = user_ans == correct_idx
                if is_correct:
                    st.markdown(f"""
                        <div class="answer-correct">
                            <div style="font-size: 18px; margin-bottom: 10px;">
                                ✅ <strong>Correct!</strong>
                            </div>
                            <div style="font-size: 15px;">
                                The answer is <strong>{correct_letter}. {correct_text}</strong>
                            </div>
                            <div style="font-size: 14px; margin-top: 12px; opacity: 0.95;">
                                💡 <em>{explanation}</em>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    user_letter = chr(65 + user_ans)
                    user_text = choices[user_ans] if user_ans < len(choices) else ""
                    st.markdown(f"""
                        <div class="answer-wrong">
                            <div style="font-size: 18px; margin-bottom: 10px;">
                                ❌ <strong>Incorrect</strong>
                            </div>
                            <div style="font-size: 15px;">
                                You selected <strong>{user_letter}. {user_text}</strong><br>
                                Correct answer: <strong>{correct_letter}. {correct_text}</strong>
                            </div>
                            <div style="font-size: 14px; margin-top: 12px; opacity: 0.95;">
                                💡 <em>{explanation}</em>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="answer-neutral">
                        <div style="font-size: 18px; margin-bottom: 10px;">
                            📖 <strong>Answer</strong>
                        </div>
                        <div style="font-size: 15px;">
                            <strong>{correct_letter}. {correct_text}</strong>
                        </div>
                        <div style="font-size: 14px; margin-top: 12px; opacity: 0.95;">
                            💡 <em>{explanation}</em>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
    
    # Score and control buttons
    st.markdown("### 📊 Quiz Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔓 Reveal All", key="quiz_reveal_all", use_container_width=True):
            st.session_state.revealed_answers = set(range(len(questions)))
            st.rerun()
    
    with col2:
        if st.button("🔒 Hide All", key="quiz_hide_all", use_container_width=True):
            st.session_state.revealed_answers = set()
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Quiz", key="quiz_reset", use_container_width=True):
            st.session_state.revealed_answers = set()
            st.session_state.user_answers = {}
            st.rerun()
    
    # Calculate and show score if any answers revealed
    answered_count = len([q for q in range(len(questions)) if q in st.session_state.user_answers and st.session_state.user_answers[q] is not None])
    revealed_count = len(st.session_state.revealed_answers)
    
    if revealed_count > 0 and answered_count > 0:
        correct_count = sum(
            1 for q_idx, q in enumerate(questions)
            if q_idx in st.session_state.revealed_answers and 
               st.session_state.user_answers.get(q_idx) == q.get("answer_index", 0)
        )
        answered_and_revealed = len([q for q in st.session_state.revealed_answers if st.session_state.user_answers.get(q) is not None])
        
        if answered_and_revealed > 0:
            percentage = (correct_count / answered_and_revealed) * 100
            
            if percentage >= 80:
                color = "#11998e"
                emoji = "🎉"
                message = "Excellent work!"
            elif percentage >= 60:
                color = "#f7971e"
                emoji = "👍"
                message = "Good job!"
            else:
                color = "#eb3349"
                emoji = "📚"
                message = "Keep studying!"
            
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color} 0%, {color}cc 100%);
                    color: white;
                    border-radius: 15px;
                    padding: 25px;
                    text-align: center;
                    margin: 25px 0;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
                ">
                    <div style="font-size: 28px; margin-bottom: 10px;">{emoji} Score: {correct_count}/{answered_and_revealed} ({percentage:.0f}%)</div>
                    <div style="font-size: 18px; opacity: 0.95;">{message}</div>
                </div>
            """, unsafe_allow_html=True)


def render_interactive_mindmap(mindmap_data):
    """Render mindmap as a clean, organized visual diagram."""
    import json
    import re
    
    raw_text = mindmap_data.strip()
    title = "Your Notes"
    branches = []
    
    # Parse JSON - extract ONLY the first valid JSON object
    try:
        data = json.loads(raw_text)
        title = data.get("title", "Your Notes")
        for branch in data.get("branches", []):
            name = branch.get("name", "")
            items = branch.get("items", [])
            # Clean the name - remove any JSON artifacts
            name = re.sub(r'^["\']+|["\']+$', '', str(name)).strip()
            if name and not name.startswith('"') and ":" not in name[:5]:
                clean_items = []
                for item in items:
                    item_str = re.sub(r'^["\']+|["\']+$', '', str(item)).strip()
                    if item_str:
                        clean_items.append(item_str)
                branches.append({"name": name, "items": clean_items})
    except:
        pass
    
    # Limit to 6 branches max for cleaner display
    branches = branches[:6]
    
    # If no branches, show error
    if not branches:
        st.warning("⚠️ Couldn't generate mindmap. The content may not have enough structure.")
        with st.expander("View raw response"):
            st.code(raw_text)
        return
    
    # Colors for branches
    colors = [
        "#667eea", "#11998e", "#fc4a1a", 
        "#ee0979", "#4facfe", "#f093fb"
    ]
    
    # Clean, simple CSS
    st.markdown("""
        <style>
        .mindmap-box {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 20px;
            padding: 25px;
            margin: 15px 0;
        }
        .mindmap-title {
            text-align: center;
            color: #fff;
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 25px;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 30px;
            display: inline-block;
            width: 100%;
            box-sizing: border-box;
        }
        .mindmap-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .mindmap-card {
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            overflow: hidden;
        }
        .mindmap-card-title {
            color: #fff;
            padding: 12px 15px;
            font-weight: 600;
            font-size: 14px;
        }
        .mindmap-card-items {
            padding: 12px 15px;
            background: rgba(0,0,0,0.2);
        }
        .mindmap-card-item {
            color: rgba(255,255,255,0.9);
            font-size: 13px;
            padding: 6px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .mindmap-card-item:last-child {
            border-bottom: none;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Build clean HTML
    html = ['<div class="mindmap-box">']
    html.append(f'<div class="mindmap-title">🧠 {title}</div>')
    html.append('<div class="mindmap-grid">')
    
    for i, branch in enumerate(branches):
        color = colors[i % len(colors)]
        html.append(f'<div class="mindmap-card">')
        html.append(f'<div class="mindmap-card-title" style="background: {color};">📌 {branch["name"]}</div>')
        
        items = branch.get("items", [])
        if items:
            html.append('<div class="mindmap-card-items">')
            for item in items[:5]:  # Max 5 items per branch
                html.append(f'<div class="mindmap-card-item">• {item}</div>')
            html.append('</div>')
        
        html.append('</div>')
    
    html.append('</div>')
    html.append('</div>')
    
    st.markdown(''.join(html), unsafe_allow_html=True)
    
    # Simple stats
    total_items = sum(len(b.get("items", [])[:5]) for b in branches)
    st.caption(f"📊 {len(branches)} topics, {total_items} details")
    
    with st.expander("📄 View Raw Output"):
        st.code(raw_text)


def render_interactive_flashcards(flashcards_data):
    """Render flashcards as interactive click-to-reveal cards."""
    
    # Get flashcards list from data
    cards = []
    if isinstance(flashcards_data, dict):
        cards = flashcards_data.get("flashcards", [])
    elif isinstance(flashcards_data, list):
        cards = flashcards_data
    
    if not cards:
        # Show error message if available
        if isinstance(flashcards_data, dict):
            if "error" in flashcards_data:
                st.error(f"⚠️ {flashcards_data['error']}")
            raw = flashcards_data.get("_raw") or flashcards_data.get("raw")
            if raw:
                with st.expander("🔍 View raw model response (for debugging)"):
                    st.code(raw)
        st.warning("No flashcards generated. Make sure you have uploaded documents first, then try again.")
        return
    
    # Initialize session state for flipped cards
    if "flipped_cards" not in st.session_state:
        st.session_state.flipped_cards = set()
    
    # CSS for flashcard styling
    st.markdown("""
        <style>
        .flashcard-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            padding: 20px 0;
        }
        .flashcard {
            width: 280px;
            min-height: 180px;
            perspective: 1000px;
            cursor: pointer;
        }
        .flashcard-inner {
            position: relative;
            width: 100%;
            min-height: 180px;
            text-align: center;
            transition: transform 0.6s;
            transform-style: preserve-3d;
        }
        .flashcard.flipped .flashcard-inner {
            transform: rotateY(180deg);
        }
        .flashcard-front, .flashcard-back {
            position: absolute;
            width: 100%;
            min-height: 180px;
            backface-visibility: hidden;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            box-sizing: border-box;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .flashcard-front {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 16px;
            font-weight: 500;
        }
        .flashcard-back {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            transform: rotateY(180deg);
            font-size: 15px;
        }
        .flashcard-number {
            position: absolute;
            top: 10px;
            left: 15px;
            font-size: 12px;
            opacity: 0.8;
        }
        .click-hint {
            position: absolute;
            bottom: 10px;
            font-size: 11px;
            opacity: 0.7;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Instructions
    st.markdown("### 🎴 Study Flashcards")
    st.caption("Click 'Show Answer' to reveal the definition or answer!")
    
    # Display cards in a list format for better readability
    for card_idx, card in enumerate(cards):
        front = card.get("front", "Question")
        back = card.get("back", "Answer")
        
        card_key = f"card_{card_idx}"
        is_flipped = card_idx in st.session_state.flipped_cards
        
        # Question card (always visible)
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 15px;
                padding: 20px 25px;
                margin: 15px 0 10px 0;
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
            ">
                <div style="font-size: 11px; opacity: 0.8; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">
                    📚 Flashcard {card_idx + 1} of {len(cards)}
                </div>
                <div style="font-size: 17px; font-weight: 600; line-height: 1.5;">
                    {front}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Show/Hide Answer button
        btn_label = "🙈 Hide Answer" if is_flipped else "👁️ Show Answer"
        if st.button(btn_label, key=card_key, type="secondary" if is_flipped else "primary"):
            if is_flipped:
                st.session_state.flipped_cards.discard(card_idx)
            else:
                st.session_state.flipped_cards.add(card_idx)
            st.rerun()
        
        # Answer card (only visible when flipped)
        if is_flipped:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    color: white;
                    border-radius: 15px;
                    padding: 20px 25px;
                    margin: 10px 0 20px 0;
                    box-shadow: 0 6px 20px rgba(17, 153, 142, 0.3);
                ">
                    <div style="font-size: 11px; opacity: 0.8; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">
                        ✅ Answer
                    </div>
                    <div style="font-size: 16px; line-height: 1.6;">
                        {back}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.divider()
    
    # Control buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Show All Answers", key="flashcard_show_all", use_container_width=True):
            st.session_state.flipped_cards = set(range(len(cards)))
            st.rerun()
    with col2:
        if st.button("🔒 Hide All Answers", key="flashcard_hide_all", use_container_width=True):
            st.session_state.flipped_cards = set()
            st.rerun()
    with col3:
        if st.button("🎲 Shuffle Cards", key="flashcard_shuffle", use_container_width=True):
            import random
            random.shuffle(cards)
            st.session_state.flipped_cards = set()
            st.session_state.current_flashcards = {"flashcards": cards}
            st.rerun()


def get_note_options(include_all=False, include_none=False):
    """Get fresh note options for dropdowns."""
    state = load_notes()
    notes = state.get("notes", {})
    note_map = {nid: n["title"] for nid, n in notes.items()}
    
    options = []
    if include_all:
        options.append("📚 All Notes")
    if include_none:
        options.append("➖ No specific note")
    
    for nid, title in note_map.items():
        options.append(f"📝 {title} ({nid})")
    
    return options, notes


def extract_note_id(selection):
    """Extract note_id from a dropdown selection."""
    if not selection or "All Notes" in selection or "No specific" in selection:
        return None
    if "(" in selection and ")" in selection:
        return selection.split("(")[-1].rstrip(")")
    return None


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

    tabs = st.tabs(["📤 Upload", "💬 Chat", "📝 Quizzes", "📋 Summarize", "🎴 Flashcards", "🧠 Mindmap"]) 

    # Upload tab
    with tabs[0]:
        st.header("📤 Upload Your Study Materials")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded = st.file_uploader("Select files to upload", accept_multiple_files=True)
            
            if st.button("📤 Upload & Index", type="primary"):
                if not uploaded:
                    st.warning("No files selected")
                else:
                    res = ingest_files(uploaded)
                    if res.get("ok"):
                        st.success(f"✅ Indexed {res['docs']} documents into {res['chunks']} chunks!")
                        st.balloons()
                    else:
                        st.error(res.get("message") or res.get("error"))
        
        with col2:
            st.markdown("""
                <div class='card'>
                    <b>💡 How it works:</b><br><br>
                    1. Upload your files here<br>
                    2. Go to any tab to study!<br>
                    3. Chat, Quiz, Summarize, etc.<br><br>
                    <b>Supported formats:</b><br>
                    PDF, DOCX, TXT, Images<br><br>
                    <b>Note:</b> Uploading new files replaces old content automatically.
                </div>
            """, unsafe_allow_html=True)

    # Chat tab
    with tabs[1]:
        st.header("💬 Chat with your documents")
        
        question = st.text_input("Ask a question about your uploaded materials")
        if st.button("🔍 Ask", type="primary"):
            if not question.strip():
                st.warning("Enter a question")
            else:
                with st.spinner("Thinking..."):
                    answer = ask_question(question.strip())
                st.success("Answer")
                st.write(answer)

    # Quizzes tab
    with tabs[2]:
        st.header("📝 Interactive Quiz")
        st.caption("Test your knowledge — answer questions and reveal the correct answers!")
        
        topic = st.text_input("Topic (optional)", key="quiz_topic")
        n = st.number_input("Number of questions", min_value=1, max_value=50, value=5)
        
        if st.button("Generate Quiz", type="primary"):
            with st.spinner("Creating your quiz..."):
                out = generate_quiz(topic, n)
            st.session_state.current_quiz = out
            st.session_state.revealed_answers = set()
            st.session_state.user_answers = {}
            st.rerun()
        
        if "current_quiz" in st.session_state and st.session_state.current_quiz:
            render_interactive_quiz(st.session_state.current_quiz)

    # Summarize tab
    with tabs[3]:
        st.header("📋 Summarize")
        st.caption("Get a concise summary of your study materials")
        
        topic = st.text_input("Topic to summarize (optional)", key="summary_topic")
        
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Summarizing..."):
                out = generate_summary(topic)
            st.write(out)

    # Flashcards tab
    with tabs[4]:
        st.header("🎴 Interactive Flashcards")
        st.caption("Click on cards to reveal answers — test your knowledge!")
        
        topic = st.text_input("Topic (optional)", key="flash_topic")
        n = st.number_input("Number of flashcards", min_value=1, max_value=50, value=10, key="flash_n")
        
        if st.button("Generate Flashcards", type="primary"):
            with st.spinner("Creating your flashcards..."):
                out = generate_flashcards(topic, int(n))
            st.session_state.current_flashcards = out
            st.session_state.flipped_cards = set()
            st.rerun()
        
        if "current_flashcards" in st.session_state and st.session_state.current_flashcards:
            render_interactive_flashcards(st.session_state.current_flashcards)

    # Mindmap tab
    with tabs[5]:
        st.header("🧠 Interactive Mindmap")
        st.caption("Visualize your documents as an interactive concept map!")
        
        topic = st.text_input("Central topic (optional)", key="mind_topic")
        
        if st.button("Generate Mindmap", type="primary"):
            with st.spinner("Creating your mindmap..."):
                out = generate_mindmap(topic)
            st.session_state.current_mindmap = out
            st.rerun()
        
        if "current_mindmap" in st.session_state and st.session_state.current_mindmap:
            render_interactive_mindmap(st.session_state.current_mindmap)


if __name__ == "__main__":
    main()
