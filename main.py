import streamlit as st
import sqlite3
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- SQLite DB setup ---
DB_FILE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

def load_history_from_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_history ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def add_message_to_db(role, content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

# Initialize the database
init_db()

def build_context(chat_history, limit=5):
    # Use only the last few messages as context to avoid exceeding token limits
    recent = chat_history[-limit:]
    context_lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_lines.append(f"{role}: {msg['content']}")
    return "\n".join(context_lines)

# --- Streamlit page configuration ---
st.set_page_config(page_title="Chat with Injamul", layout="wide")

# Get your secret API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- Document loading & splitting ---
file_path = 'who_am_I.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()
full_text = "\n".join([d.page_content for d in docs])
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text(full_text)
documents = [Document(page_content=chunk) for chunk in chunks]

# --- Embeddings & vector store ---
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embedding)
retriever = vector_store.as_retriever()

# --- Prompt template with previous conversation context ---
prompt = ChatPromptTemplate([
    """
    You have to act like Injamul. Your bio is provided in the context. People will ask questions 
    and you should answer based only on the provided context and previous conversation.
    
    <previous_conversation>
    {previous_conversation}
    <previous_conversation>
    
    <context>
    {context}
    <context>
    
    Question: {input}
    Answer:
    """
])

# --- Initialize chat history from SQLite DB ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history_from_db()

# --- Sidebar: Display chat history with clickable expanders ---
with st.sidebar:
    st.header("Chat History")
    for i, msg in enumerate(st.session_state.chat_history):
        # Display a preview of the message (first 50 characters)
        preview = msg["content"][:50] + ("..." if len(msg["content"]) > 50 else "")
        expander_title = f"{msg['role'].capitalize()}: {preview}"
        with st.expander(expander_title):
            st.write(msg["content"])

# --- Main UI ---
st.title("💬 Chat with Injamul")
model_choice = st.sidebar.selectbox(
    "Select a model for responses:",
    [
      "llama-3.1-8b-instant",
      "gemma2-9b-it",
      "deepseek-r1-distill-llama-70b",
      "deepseek-r1-distill-qwen-32b",
      "qwen-2.5-32b"
    ]
)

# Display conversation in the main chat window
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
input_query = st.chat_input("Type your message…")
if input_query:
    # Append user message to session state and save in DB
    st.session_state.chat_history.append({"role": "user", "content": input_query})
    add_message_to_db("user", input_query)
    with st.chat_message("user"):
        st.markdown(input_query)
    
    # Build previous conversation context for the prompt
    previous_context = build_context(st.session_state.chat_history)
    
    # Initialize the LLM with your API key and selected model
    llm = ChatGroq(model=model_choice, api_key=GROQ_API_KEY)
    
    # Build chains with context included
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    
    # Invoke the chain and get the answer (passing additional context)
    result = retrieval_chain.invoke({
        "input": input_query,
        "previous_conversation": previous_context,
        "context": "\n".join([d.page_content for d in documents]),
    })
    answer = result["answer"].split("</think>")[-1].strip()
    
    # Append assistant message to session state and save in DB
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    add_message_to_db("assistant", answer)
    with st.chat_message("assistant"):
        st.markdown(answer)
