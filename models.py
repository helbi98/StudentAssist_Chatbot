import os
from dotenv import load_dotenv

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

load_dotenv()

# Default constants
DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_COLLECTION = "university_docs"
DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL = "llama-3.3-70b-versatile"  
DEFAULT_TEMP = 0.2

def _construct_chatgroq(model_name: str, temperature: float, groq_api_key: str):
    return ChatGroq(model=model_name, temperature=temperature, groq_api_key=groq_api_key)

def create_retriever_and_llm(
    vectorstore=None,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    k: int = 8,
    model_name: str | None = None,
    temperature: float = DEFAULT_TEMP,
):

    # 1) ensure vectorstore
    if vectorstore is None:
        embedding_model = SentenceTransformerEmbeddings(model_name=DEFAULT_EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=chroma_dir,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

    # 2) create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 3) create ChatGroq LLM
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY not found in environment. Set it in .env or environment variables.")

    if model_name is None:
        model_name = DEFAULT_MODEL

    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string.")

    llm = _construct_chatgroq(model_name=model_name, temperature=temperature, groq_api_key=groq_api_key)

    return retriever, llm
