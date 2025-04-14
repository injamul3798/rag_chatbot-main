import os
import time
import uuid
import sqlite3
from datetime import datetime, timedelta

import streamlit as st
st.set_page_config(page_title="💬 Chat with Injamul", layout="wide")  # Must be at the very top

from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader  # For PDF processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document

load_dotenv()
DB_FILE = "chat_history.db"

# --- Helper functions for time ---
def current_time():
    return datetime.now() + timedelta(hours=6)

def current_time_str(fmt="%Y-%m-%d %H:%M:%S"):
    return current_time().strftime(fmt)

def current_time_short():
    return current_time().strftime("%H:%M:%S")

# --- DATABASE UTILITIES ---
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            title TEXT,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()

def load_conversations(user_id):
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        "SELECT id, title FROM conversations WHERE user_id = ? ORDER BY id DESC",
        (user_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1]} for r in rows]

def create_new_conversation(user_id, title=None):
    if not title:
        title = "Untitled Chat"
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    now = current_time_str()
    c.execute(
        "INSERT INTO conversations (user_id, title, created_at) VALUES (?, ?, ?)",
        (user_id, title, now)
    )
    conv_id = c.lastrowid
    conn.commit()
    conn.close()
    return conv_id

def delete_conversation(conv_id):
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conv_id,))
    c.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()

def load_messages(conv_id):
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        "SELECT role, content, timestamp FROM chat_messages WHERE conversation_id = ? ORDER BY id ASC",
        (conv_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

def add_message(conv_id, role, content):
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    now = current_time_str()
    c.execute(
        "INSERT INTO chat_messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (conv_id, role, content, now)
    )
    conn.commit()
    conn.close()

def push_to_github():
    os.system(f"git add {DB_FILE}")
    os.system('git commit -m "Update conversation data"')
    os.system("git push")

# --- INITIALIZE & IDENTIFY USER ---
init_db()
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# --- FUNCTIONS FOR PROCESSING UPLOADED FILES ---
def process_uploaded_file(uploaded_file):
    """
    Processes an uploaded file (PDF or text) and returns its content as a string.
    """
    if uploaded_file is not None:
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            loader = PyPDFLoader(uploaded_file)
            docs = loader.load()
            content = "\n".join(d.page_content for d in docs)
            return content
        elif file_type.startswith("text"):
            return uploaded_file.read().decode("utf-8")
    return None

# --- SIDE BAR: Conversation and File Upload Section ---
st.sidebar.title("")
user_id = st.session_state.user_id

# File upload option
uploaded_file = st.sidebar.file_uploader("Upload a file for context (PDF or TXT)", type=["pdf", "txt"])
# Process file if uploaded; else use default context
uploaded_file_text = process_uploaded_file(uploaded_file)
if uploaded_file_text:
    st.sidebar.success("File uploaded successfully!")

# Load user's conversations
convs = load_conversations(user_id)
if not convs:
    first_id = create_new_conversation(user_id, "Chat " + current_time_short())
    st.session_state.current_conv = first_id
    convs = load_conversations(user_id)

def shorten_title(title, max_length=30):
    return title if len(title) <= max_length else title[:max_length-3] + "..."

# Sidebar - Conversation list
conversation_options = {shorten_title(conv["title"]): conv["id"] for conv in convs}
selected_title = st.sidebar.selectbox("Conversations", list(conversation_options.keys()))
st.session_state.current_conv = conversation_options[selected_title]

if st.sidebar.button("🗑️ Delete Selected"):
    delete_conversation(st.session_state.current_conv)
    convs = load_conversations(user_id)
    if convs:
        st.session_state.current_conv = convs[0]["id"]
    else:
        st.session_state.current_conv = create_new_conversation(user_id, "Chat " + current_time_short())
    conversation_options = {shorten_title(conv["title"]): conv["id"] for conv in convs}

with st.sidebar.expander("➕ New Conversation"):
    custom_title = st.text_input("Enter chat title", key="new_chat_title")
    if st.button("Create Chat", key="create_new_conv"):
        new_title = custom_title.strip() if custom_title.strip() else "Chat " + current_time_short()
        new_id = create_new_conversation(user_id, new_title)
        st.session_state.current_conv = new_id

# --- VECTOR STORE SETUP --- 
@st.cache_resource
def build_retriever(file_text):
    # Use the uploaded file content if available; otherwise, load from default file.
    if file_text:
        text = file_text
    else:
        # Pass the file path directly to PyPDFLoader to avoid file handle errors.
        loader = PyPDFLoader("who_am_I.pdf")
        docs = loader.load()
        text = "\n".join(d.page_content for d in docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(documents, embed)
    return store.as_retriever(), documents

retriever, all_docs = build_retriever(uploaded_file_text)

# --- PROMPT TEMPLATE ---
prompt = ChatPromptTemplate([
    """
    You have to act like Injamul. Your bio is provided in the context.
    Answer questions based only on the provided context and previous conversation.

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

# --- MAIN AREA: Chat Display ---
st.title("💬 Chat with Injamul")
model_choice = st.selectbox("Model for responses:", [
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "qwen-2.5-32b",
    "whisper-large-v3",
])

# Load & display chat history for the selected conversation
history = load_messages(st.session_state.current_conv)
st.session_state.chat_history = history

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

instruction = "Type your message…" if not uploaded_file_text else "Type your message or ask a question about the uploaded file…"
user_input = st.chat_input(instruction)

if user_input:
    add_message(st.session_state.current_conv, "user", user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    recent = st.session_state.chat_history[-5:]
    prev_ctx = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)
    
    combined_context = "\n".join(d.page_content for d in all_docs)
    
    llm = ChatGroq(model=model_choice, api_key=st.secrets["GROQ_API_KEY"])
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    result = chain.invoke({
        "input": user_input,
        "previous_conversation": prev_ctx,
        "context": combined_context,
    })
    answer = result["answer"].split("</think>")[-1].strip()
    
    add_message(st.session_state.current_conv, "assistant", answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    push_to_github()
