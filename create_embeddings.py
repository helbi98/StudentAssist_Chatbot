import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

CHUNKS_FILE = "chunks.jsonl"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "university_docs"

def load_chunks(file_path):
    """Load chunks from the JSONL file produced by prepare_chunks.py"""
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    print("Loading chunks from file...")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks.")

    # Convert chunks into LangChain Document objects
    docs = [
        Document(page_content=c["text"], metadata={"source": c["source"], "id": c["id"]})
        for c in chunks
    ]

    print("Initializing multilingual embedding model...")
    embedding_model = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    print("Creating ChromaDB collection and storing embeddings...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

    vectordb.persist()
    print("Embeddings created and stored successfully in:", CHROMA_DIR)

if __name__ == "__main__":
    main()
