import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from models import create_retriever_and_llm, DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Initialize pipeline (retriever + llm)
RETRIEVER, LLM = create_retriever_and_llm(
    vectorstore=None,
    chroma_dir=DEFAULT_CHROMA_DIR,
    collection_name=DEFAULT_COLLECTION,
    k=5,
    temperature=0.2,
)

PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
You are StudentAssist Bot, a friendly and knowledgeable university assistant for MSc Data and Knowledge Engineering students.

You are given a context delimited by <context></context> along with a user question.
Your objective is to generate an appropriate answer using ONLY the information in the context.

- Include actual website links (that contain https://) and contact details if they are mentioned in the context.
- Only include website links found in the database.
- Do NOT fabricate any information or URLs.
- Format the answer neatly and clearly.
- Remember the conversation flow and respond naturally as in a chat.
- Answer in the same language as the user (English or German).

<context>
{context}
</context>

Chat History:
{chat_history}

Question: {question}

Answer:
"""
)

# Memory (so the bot remembers prior exchanges)
MEMORY = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

MEMORY.output_key = "answer"
# ------------------------------

# Build the conversational retrieval chain.
CHATBOT_CHAIN = ConversationalRetrievalChain.from_llm(
    llm=LLM,
    retriever=RETRIEVER,
    memory=MEMORY,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    return_source_documents=True,
)

CHATBOT_CHAIN.output_key = "answer"
# ---------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        result = CHATBOT_CHAIN.invoke({"question": question})
    except TypeError:
        result = CHATBOT_CHAIN({"question": question})

    answer = result.get("answer") or result.get("result") or ""
    if isinstance(answer, list):
        # if memory/chain returns messages, join them
        answer = " ".join([str(a) for a in answer])

    answer = str(answer).strip()
    source_docs = result.get("source_documents", [])

    sources = []
    for d in source_docs:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source") or meta.get("url")
        if src and isinstance(src, str) and src.startswith("http") and src not in sources:
            sources.append(src)

    return jsonify({"answer": answer, "sources": sources})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
