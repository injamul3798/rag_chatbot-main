import os
import time
import uuid
import sqlite3
from datetime import datetime

import streamlit as st
st.set_page_config(page_title="💬 Chat with Injamul", layout="wide")  # Must be at the very top

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
    # Return conversation details; no need to display the conversation ID.
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

def create_new_conversation(user_id, title=None):
    if not title:
        title = "Untitled Chat"
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

# --- CUSTOM STYLES ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
        .stSidebar {
            background-color: #111;
            color: #ccc;
        }
        .chat-title {
            padding: 8px 10px;
            background: #333;
            border-radius: 6px;
            margin-bottom: 5px;
        }
        .chat-title button {
            background: none;
            border: none;
            color: white;
            font-weight: bold;
            width: 100%;
            text-align: left;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# --- STREAMLIT UI & LOGIC ---
st.sidebar.title("💬 Your Conversations")
user_id = st.session_state.user_id

# Load user's conversations
convs = load_conversations(user_id)
if not convs:
    # Create a first conversation if none exist
    first_id = create_new_conversation(user_id, "Chat " + time.strftime("%H:%M:%S"))
    st.session_state.current_conv = first_id
    convs = load_conversations(user_id)

# Sidebar - Conversation list using buttons
st.sidebar.markdown("### Conversations")
for conv in convs:
    # Display the title and creation time; omit conversation ID.
    chat_display = f"{conv['title']} ({conv['created_at']})"
    if st.sidebar.button(chat_display, key=f"conv_{conv['id']}"):
        st.session_state.current_conv = conv['id']

# Option to delete the currently selected conversation
if st.sidebar.button("🗑️ Delete Selected Conversation"):
    delete_conversation(st.session_state.current_conv)
    convs = load_conversations(user_id)
    if convs:
        st.session_state.current_conv = convs[0]["id"]
    else:
        st.session_state.current_conv = create_new_conversation(user_id, "Chat " + time.strftime("%H:%M:%S"))

# Sidebar - New Conversation with custom title
with st.sidebar.expander("➕ New Conversation"):
    custom_title = st.text_input("Enter chat title", key="new_chat_title")
    if st.button("Create Chat", key="create_new_conv"):
        new_title = custom_title.strip() if custom_title.strip() else "Chat " + time.strftime("%H:%M:%S")
        new_id = create_new_conversation(user_id, new_title)
        st.session_state.current_conv = new_id

# Load & display chat history
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

# Display chat messages in the main area
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
user_input = st.chat_input("Type your message…")
if user_input:
    # Save user message
    add_message(st.session_state.current_conv, "user", user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build previous conversation context using the last 5 messages
    recent = st.session_state.chat_history[-5:]
    prev_ctx = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

    # Call the model via ChatGroq
    llm = ChatGroq(model=model_choice, api_key=st.secrets["GROQ_API_KEY"])
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    result = chain.invoke({
        "input": user_input,
        "previous_conversation": prev_ctx,
        "context": "\n".join(d.page_content for d in all_docs),
    })
    answer = result["answer"].split("</think>")[-1].strip()

    # Save the assistant response
    add_message(st.session_state.current_conv, "assistant", answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Optionally push updates to GitHub
    push_to_github()
