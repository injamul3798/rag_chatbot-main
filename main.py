# main.py

import os
import time
import uuid
import sqlite3
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Fix for missing init_empty_weights in Transformers v5+
from transformers.utils import no_init_weights, init_empty_weights

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
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
        "SELECT id, title, created_at FROM conversations "
        "WHERE user_id = ? ORDER BY id DESC",
        (user_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

def create_new_conversation(user_id, title):
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        "SELECT role, content, timestamp FROM chat_messages "
        "WHERE conversation_id = ? ORDER BY id ASC",
        (conv_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

def add_message(conv_id, role, content):
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO chat_messages (conversation_id, role, content, timestamp) "
        "VALUES (?, ?, ?, ?)",
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


# --- VECTOR STORE SETUP ---
@st.cache_resource
def build_retriever():
    loader = PyPDFLoader("who_am_I.pdf")
    docs = loader.load()
    text = "\n".join(d.page_content for d in docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(documents, embed)
    return store.as_retriever(), documents

retriever, all_docs = build_retriever()


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


# --- STREAMLIT UI & LOGIC ---
st.set_page_config(page_title="💬 Chat with Injamul", layout="wide")
st.sidebar.title("💬 Your Conversations")

user_id = st.session_state.user_id

# Load this user's conversations
convs = load_conversations(user_id)
if not convs:
    # create a first conversation if none exist
    first_id = create_new_conversation(user_id, "Chat " + time.strftime("%H:%M:%S"))
    st.session_state.current_conv = first_id
    convs = load_conversations(user_id)

# Build options for selectbox
options = [f"{c['id']}: {c['title']} ({c['created_at']})" for c in convs]
default_idx = next((i for i, c in enumerate(convs) if c["id"] == st.session_state.get("current_conv")), 0)

# Select a conversation
selected = st.sidebar.selectbox("Select Conversation", options, index=default_idx)
if selected:
    sel_id = int(selected.split(":")[0])
    st.session_state.current_conv = sel_id

# Delete button for the selected conversation
if st.sidebar.button("🗑️ Delete Selected"):
    delete_conversation(st.session_state.current_conv)
    convs = load_conversations(user_id)
    if convs:
        st.session_state.current_conv = convs[0]["id"]
    else:
        st.session_state.current_conv = create_new_conversation(user_id, "Chat " + time.strftime("%H:%M:%S"))

# New conversation button
if st.sidebar.button("➕ New Conversation"):
    new_id = create_new_conversation(user_id, "Chat " + time.strftime("%H:%M:%S"))
    st.session_state.current_conv = new_id

# Load & display history
history = load_messages(st.session_state.current_conv)
st.session_state.chat_history = history

st.title("💬 Chat with Injamul")
model_choice = st.selectbox("Model for responses:", [
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "qwen-2.5-32b"
])

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
user_input = st.chat_input("Type your message…")
if user_input:
    add_message(st.session_state.current_conv, "user", user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    recent = st.session_state.chat_history[-5:]
    prev_ctx = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

    llm = ChatGroq(model=model_choice, api_key=st.secrets["GROQ_API_KEY"])
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    result = chain.invoke({
        "input": user_input,
        "previous_conversation": prev_ctx,
        "context": "\n".join(d.page_content for d in all_docs),
    })
    answer = result["answer"].split("</think>")[-1].strip()

    add_message(st.session_state.current_conv, "assistant", answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    push_to_github()
