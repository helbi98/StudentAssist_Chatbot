# StudentAssist – University Course Chatbot

This is an AI-powered question-answering system designed to help new students get information about MSc Data & Knowledge Engineering at OVGU Magdeburg.  
It uses a **Retrieval-Augmented Generation (RAG) pipeline** with **ChromaDB** for semantic search, **SentenceTransformer embeddings** for text representation, and **Groq API-powered LLMs** for answer generation.

---

## Features
- **RAG pipeline** with context-aware answer generation
- **Sentence-aware chunking** of scraped course webpages for better context preservation
- **Semantic search** using SentenceTransformer embeddings stored in persistent ChromaDB
- **Multi-language support** – answers in English or German depending on the question
- **Conversational memory** – remembers prior exchanges in chat sessions
- **Source citation** – includes URLs of the pages used to answer questions
- **Web-based UI** – interactive chatbot interface using Flask

---

## Architecture

### 1. Data Collection
- Web scraping of OVGU MSc Data & Knowledge Engineering course pages using `scrape_university.py`.
- Extracts visible text, removes scripts, headers, footers, and unnecessary elements.

### 2. Chunking & Preprocessing
- `prepare_chunks.py` splits scraped pages into sentence-aware chunks for better retrieval.
- Chunks stored in JSONL format with metadata (source URL, chunk ID).

### 3. Vector Store
- Chunks embedded using **SentenceTransformers** (`paraphrase-multilingual-MiniLM-L12-v2`).
- Stored in a persistent **ChromaDB collection** for fast semantic similarity search.

### 4. Retrieval & RAG
- Incoming queries are processed by the LLM pipeline.
- Top-k most relevant chunks retrieved from ChromaDB.
- **ChatGroq LLM** generates context-aware answers using the retrieved chunks.

### 5. Conversational Interface
- **Flask web app** (`app.py`) provides a chat interface.
- **Memory** remembers previous user messages and bot responses.
- Answers include **source URLs** when available.

---

## Steps to run code

```bash
setx GROQ_API_KEY "your_api_key_here"
python scrape_university.py
python prepare_chunks.py
python create_embedding.py
python app.py
```
Open your browser at http://127.0.0.1:8000 to chat with StudentAssist Bot.
