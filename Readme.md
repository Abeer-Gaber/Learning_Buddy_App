# My Learning Buddy 📚

A local RAG (Retrieval-Augmented Generation) application that helps you study by uploading documents, chatting with your notes, generating quizzes, flashcards, summaries, and mindmaps.

## Features

- 📄 **Upload Documents**: Support for PDF, DOCX, and TXT files
- 💬 **Chat with Notes**: Ask questions about your uploaded documents
- 📝 **Generate Summaries**: Create study-friendly summaries with key points
- 🃏 **Flashcards**: Auto-generate flashcards from your notes
- ✅ **Quizzes**: Generate multiple-choice quizzes to test your knowledge
- 🗺️ **Mind Maps**: Visualize topics with AI-generated mind maps

## Prerequisites

- Python 3.10 or higher
- Ollama (for local LLM)

---

## Installation Guide

### Step 1: Install Ollama

Ollama is required to run the LLM locally.

#### Windows
```bash
# Download and install from the official website
Visit: https://ollama.com/download

### Step 2: Start Ollama

```bash
# Start the Ollama service
ollama serve
```

Keep this running in a separate terminal window.

### Step 3: Download Required Models

```bash
# Download the LLM model (required)
ollama pull llama3.2:latest

# Download the embedding model (recommended for better performance)
ollama pull nomic-embed-text
```

**Verify models are installed:**
```bash
ollama list
```

You should see both `llama3.2:latest` and `nomic-embed-text:latest` in the list.

### Step 4: Clone or Download This Repository

```bash
cd /path/to/your/projects
# (If using git: git clone <your-repo-url>)
cd Learning_Buddy_App
```

### Step 5: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 6: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 7: Configure Models

**Important:** Make sure the model names in your config match what you downloaded from Ollama.

#### Check your Ollama models:
```bash
ollama list
```

#### Update `core.py`:
Open `core.py` and verify these lines match your installed models:

```python
LLM_MODEL = "llama3.2:latest"  # Must match: ollama list
EMBED_MODEL = "nomic-embed-text:latest"  # Must match: ollama list
```

#### Update `app.py`:
Open `app.py` and verify these lines match your installed models:

```python
LLM_MODEL = "llama3.2:latest"  # Must match: ollama list
EMBED_MODEL = "nomic-embed-text:latest"  # Must match: ollama list
```

**Common model tags:**
- If you pulled `llama3.2` without `:latest`, check `ollama list` for the exact tag
- Example tags: `llama3.2:1b`, `llama3.2:3b`, `llama3.2:latest`
- Use the **exact tag** shown by `ollama list`

---

## Running the Application

### Option 1: Run with Streamlit (Recommended)

```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate    # On Windows

# Run the Streamlit app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Option 2: Run FastAPI Server (Alternative)

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Access the API at:
- Main page: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

---

## Troubleshooting

### Error: "Collection expecting embedding with dimension of 768, got 3072"

This happens when you change the embedding model after creating the database.

**Solution:**
```bash
# Delete the old vector database
rm -rf ./chroma_db

# Restart the application
streamlit run streamlit_app.py
```

You'll need to re-upload your documents.

### Error: "No module named 'streamlit'"

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Error: "No module named 'langchain_ollama'"

```bash
pip install langchain-ollama
```

### Error: Model not found

```bash
# Check which models you have
ollama list

# Pull the required models
ollama pull mistral
ollama pull nomic-embed-text:latest

# Update core.py and app.py to match the exact model tags from 'ollama list'
```

### Ollama connection errors

Make sure Ollama is running:
```bash
# Start Ollama (in a separate terminal)
ollama serve

# Test if it's running
curl http://127.0.0.1:11434/api/tags
```

### Port already in use

```bash
# For Streamlit (default port 8501)
streamlit run streamlit_app.py --server.port 8502

# For FastAPI (default port 8000)
uvicorn app:app --port 8001
```

---

## Usage Guide

### 1. Upload Documents

1. Go to the **Upload** tab
2. Click "Browse files" and select your PDF, DOCX, or TXT files
3. (Optional) Associate with a note for better organization
4. Click "Upload & Index"

### 2. Chat with Your Notes

1. Go to the **Chat** tab
2. (Optional) Select a specific note to query
3. Type your question
4. Click "Ask"

### 3. Generate Study Materials

**Summaries:**
- Go to **Summarize** tab
- Enter a topic (optional)
- Click "Generate Summary"

**Flashcards:**
- Go to **Flashcards** tab
- Enter topic and number of cards
- Click "Generate Flashcards"

**Quizzes:**
- Go to **Quizzes** tab
- Enter topic and number of questions
- Click "Generate Quiz"

**Mind Maps:**
- Go to **Mindmap** tab
- Enter central topic
- Click "Generate Mindmap"

---

## Project Structure

```
Learning_Buddy_App/
├── streamlit_app.py      # Streamlit UI (main interface)
├── app.py                # FastAPI server (alternative)
├── core.py               # Core utilities and LLM setup
├── requirements.txt      # Python dependencies
├── chroma_db/           # Vector database (auto-created)
├── data/                # State and uploaded files (auto-created)
│   └── state.json       # Application state
└── README.md            # This file
```

---

## Configuration

### Customize Models

Edit `core.py` to change models:

```python
# LLM Configuration
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LLM_MODEL = "llama3.2:latest"          # Change to your preferred LLM
EMBED_MODEL = "nomic-embed-text:latest" # Change to your preferred embeddings

# Vector Store Configuration
PERSIST_DIR = "./chroma_db"
COLLECTION = "study_rag"
TOP_K = 5  # Number of relevant chunks to retrieve

# Chunking Configuration (in streamlit_app.py and app.py)
chunk_size = 900      # Size of each text chunk
chunk_overlap = 150   # Overlap between chunks
```

### Available Ollama Models

Some popular models you can use:

**LLM Models:**
- `llama3.2:1b` - Smallest, fastest
- `llama3.2:3b` - Good balance
- `llama3.2:latest` - Best quality (default)
- `mistral:latest` - Alternative model
- `phi3:latest` - Microsoft's compact model

**Embedding Models:**
- `nomic-embed-text:latest` - Recommended (768 dimensions)
- `mxbai-embed-large:latest` - Alternative

To switch models:
```bash
# Pull the new model
ollama pull mistral

# Update core.py and app.py
LLM_MODEL = "mistral"

# Delete old database if changing embedding model
rm -rf ./chroma_db
```

---

## Reset Everything

If you want to start fresh:

```bash
# Delete vector database
rm -rf ./chroma_db

# Delete application state
rm -rf ./data

# Restart the app
streamlit run streamlit_app.py
```