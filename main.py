import os
import time
import sqlite3
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

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
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

def load_conversations():
    """Return all conversations ordered newest first."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

def load_messages(conv_id):
    """Return all messages for a given conversation_id."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        "SELECT role, content, timestamp FROM chat_messages "
        "WHERE conversation_id = ? ORDER BY id ASC", (conv_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

def create_new_conversation(title):
    """Insert a new conversation row and return its ID."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO conversations (title, created_at) VALUES (?, ?)", (title, now))
    conv_id = c.lastrowid
    conn.commit()
    conn.close()
    return conv_id

def add_message(conv_id, role, content):
    """Insert a new message."""
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
    """Commit & push the DB file—ensure creds are set in your environment."""
    os.system(f"git add {DB_FILE}")
    os.system('git commit -m "Update conversation data"')
    os.system("git push")


# --- INITIALIZE DB ON STARTUP ---
init_db()


# --- LOAD & PREPROCESS YOUR PDF INTO A VECTOR STORE ONCE ---
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


# --- STREAMLIT LAYOUT & LOGIC ---
st.set_page_config(page_title="💬 Chat with Injamul", layout="wide")
st.sidebar.title("💬 Conversations")

# Load conversations from DB
convs = load_conversations()
options = [f"{c['id']}: {c['title']} ({c['created_at']})" for c in convs]
if "current_conv" not in st.session_state:
    # create a default conversation if none exist
    if not convs:
        new_id = create_new_conversation("My First Chat")
        st.session_state.current_conv = new_id
    else:
        st.session_state.current_conv = convs[0]["id"]

# Sidebar: select or create new
sel = st.sidebar.selectbox("Select Conversation", options,
                           index=next(i for i,c in enumerate(convs) if c["id"]==st.session_state.current_conv))
sel_id = int(sel.split(":")[0])
if sel_id != st.session_state.current_conv:
    st.session_state.current_conv = sel_id

if st.sidebar.button("➕ New Conversation"):
    title = f"Chat {time.strftime('%H:%M:%S')}"
    new_id = create_new_conversation(title)
    st.session_state.current_conv = new_id
    # force rerun so selectbox updates
    st.experimental_rerun()

# Load messages for the active conversation
history = load_messages(st.session_state.current_conv)
if "chat_history" not in st.session_state or st.session_state.chat_history_id != st.session_state.current_conv:
    st.session_state.chat_history = history
    st.session_state.chat_history_id = st.session_state.current_conv

# Main UI
st.title("💬 Chat with Injamul")
model_choice = st.selectbox("Model for responses:", [
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "qwen-2.5-32b"
])

# Display full history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message…")
if user_input:
    # append & persist user message
    st.session_state.chat_history.append({"role":"user","content":user_input})
    add_message(st.session_state.current_conv, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build recent context (last 5 messages)
    recent = st.session_state.chat_history[-5:]
    prev_ctx = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

    # Invoke LangChain + Groq
    api_key = st.secrets["GROQ_API_KEY"]
    llm = ChatGroq(model=model_choice, api_key=api_key)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    result = chain.invoke({
        "input": user_input,
        "previous_conversation": prev_ctx,
        "context": "\n".join(d.page_content for d in all_docs),
    })
    answer = result["answer"].split("</think>")[-1].strip()

    # append & persist assistant message
    st.session_state.chat_history.append({"role":"assistant","content":answer})
    add_message(st.session_state.current_conv, "assistant", answer)

    with st.chat_message("assistant"):
        st.markdown(answer)

    # push DB to GitHub
    push_to_github()


# --- OPTIONAL: In‑Memory Fallback (no SQLite) ---
# If you’d rather not use SQLite at all, comment out the above DB code
# and replace load_messages/add_message/create_new_conversation/load_conversations
# with this simple in-memory approach:
#
# if "inmem" not in st.session_state:
#     st.session_state.inmem = {"convs": [], "msgs": {}}
# def load_conversations():
#     return [{"id":i, "title":t, "created_at":""} for i,t in enumerate(st.session_state.inmem["convs"],1)]
# def create_new_conversation(title):
#     st.session_state.inmem["convs"].append(title)
#     cid = len(st.session_state.inmem["convs"])
#     st.session_state.inmem["msgs"][cid] = []
#     return cid
# def load_messages(cid):
#     return st.session_state.inmem["msgs"].get(cid, [])
# def add_message(cid, role, content):
#     st.session_state.inmem["msgs"][cid].append({"role":role,"content":content})
# def push_to_github(): pass
#
# That will keep everything in RAM per‐session—no persistence across restarts.
