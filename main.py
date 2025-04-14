import os
import time
import uuid
import sqlite3
import tempfile
from datetime import datetime, timedelta

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

#########################################
# HELPER / UTILITY FUNCTIONS
#########################################
def current_time():
    return datetime.now() + timedelta(hours=6)

def current_time_str(fmt="%Y-%m-%d %H:%M:%S"):
    return current_time().strftime(fmt)

def current_time_short():
    return current_time().strftime("%H:%M:%S")

#########################################
# DATABASE SETUP AND FUNCTIONS
#########################################
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
    c.execute("SELECT id, title FROM conversations WHERE user_id = ? ORDER BY id DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1]} for r in rows]

def create_new_conversation(user_id, title=None):
    if not title:
        title = "Untitled Chat"
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    now = current_time_str()
    c.execute("INSERT INTO conversations (user_id, title, created_at) VALUES (?, ?, ?)", 
              (user_id, title, now))
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

#########################################
# STREAMLIT SETUP
#########################################
init_db()
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

#########################################
# FILE PROCESSING
#########################################
def process_uploaded_file(uploaded_file):
    """
    Processes an uploaded file (PDF or text) by:
    1. Saving to a temporary file
    2. Loading and reading the content
    Returns the extracted text, or None if invalid.
    """
    if uploaded_file is not None:
        file_type = uploaded_file.type.lower()
        
        # 1. Write to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file.flush()
            tmp_path = tmp_file.name
        
        # 2. Depending on file type, load from temp path
        if "pdf" in file_type:
            # Use PyPDFLoader with the path (string)
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            os.remove(tmp_path)  # remove temp file after reading
            return "\n".join(d.page_content for d in docs)
        elif "text" in file_type:
            # If it's a text file, read the file as text
            with open(tmp_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            os.remove(tmp_path)
            return file_content
        else:
            # Not a supported file type
            os.remove(tmp_path)
            return None
    return None

#########################################
# SIDEBAR - FILE UPLOAD & CONVERSATIONS
#########################################
st.sidebar.title("")
user_id = st.session_state.user_id

uploaded_file = st.sidebar.file_uploader("Upload PDF or Text for context", type=["pdf", "txt"])
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

#########################################
# VECTOR STORE SETUP
#########################################
@st.cache_resource
def build_retriever(file_text):
    """
    If user uploaded a file, use that content. Otherwise,
    load a default PDF file from path for context.
    """
    if file_text:
        text = file_text
    else:
        loader = PyPDFLoader("who_am_I.pdf")  # Provide a path string here
        docs = loader.load()
        text = "\n".join(d.page_content for d in docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(documents, embed)
    return store.as_retriever(), documents

retriever, all_docs = build_retriever(uploaded_file_text)

#########################################
# PROMPT TEMPLATE
#########################################
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

#########################################
# MAIN PAGE - CHAT DISPLAY
#########################################
st.title("💬 Chat with Injamul")

model_choice = st.selectbox("Model for responses:", [
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "qwen-2.5-32b",
    "whisper-large-v3",
])

history = load_messages(st.session_state.current_conv)
st.session_state.chat_history = history

# Display existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input prompt
prompt_text = "Type your message…" if not uploaded_file_text else "Type your message or ask about the uploaded file…"
user_input = st.chat_input(prompt_text)

if user_input:
    add_message(st.session_state.current_conv, "user", user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build short context from last few messages
    recent_msgs = st.session_state.chat_history[-5:]
    prev_ctx = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent_msgs)

    # Combine all document content
    combined_context = "\n".join(d.page_content for d in all_docs)

    # Use LLM to generate the answer
    llm = ChatGroq(model=model_choice, api_key=st.secrets["GROQ_API_KEY"])
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)
    result = chain.invoke({
        "input": user_input,
        "previous_conversation": prev_ctx,
        "context": combined_context,
    })
    raw_answer = result["answer"]

    # In some LLM outputs, a "<think>" tag may appear. 
    # If so, we remove any chain-of-thought text by splitting on "</think>"
    answer = raw_answer.split("</think>")[-1].strip()

    # Save assistant's message
    add_message(st.session_state.current_conv, "assistant", answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    push_to_github()
