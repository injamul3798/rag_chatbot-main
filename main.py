import streamlit as st
import sqlite3
import os
import time
from datetime import datetime
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

# === SQLite DB Setup ===
DB_FILE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": row[0], "title": row[1], "created_at": row[2]} for row in rows]

def load_messages(conversation_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content, timestamp FROM chat_messages WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in rows]

def create_new_conversation(title):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO conversations (title, created_at) VALUES (?, ?)", (title, timestamp))
    conn.commit()
    conversation_id = c.lastrowid
    conn.close()
    return conversation_id

def add_message(conversation_id, role, content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO chat_messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, timestamp)
    )
    conn.commit()
    conn.close()

def push_to_github():
    os.system("git add " + DB_FILE)
    os.system('git commit -m "Update conversation data"')
    os.system("git push")

# Initialize the database
init_db()

# === Document Loading & Preprocessing ===
file_path = 'who_am_I.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()
full_text = "\n".join([d.page_content for d in docs])
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text(full_text)
documents = [Document(page_content=chunk) for chunk in chunks]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embedding)
retriever = vector_store.as_retriever()

# === Prompt Template (with conversation context) ===
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

# === Session State Setup ===
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = create_new_conversation("New Conversation " + time.strftime("%H:%M:%S"))
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = load_messages(st.session_state.current_conversation_id)

# Define a function to render the app, so we can call it when needed.
def render_app():
    st.sidebar.title("ChatGPT Conversations")
    conversations = load_conversations()
    conversation_options = []
    for conv in conversations:
        conversation_options.append(f"{conv['id']}: {conv['title']} ({conv['created_at']})")

    default_idx = 0
    for idx, conv in enumerate(conversations):
        if conv["id"] == st.session_state.current_conversation_id:
            default_idx = idx
            break

    selected_conv = st.sidebar.selectbox("Select Conversation", conversation_options, index=default_idx)
    selected_conv_id = int(selected_conv.split(":")[0])
    if selected_conv_id != st.session_state.current_conversation_id:
        st.session_state.current_conversation_id = selected_conv_id
        st.session_state.chat_messages = load_messages(selected_conv_id)
        # Since we don't have experimental_rerun, call render_app() manually
        render_app()
        return

    if st.sidebar.button("New Conversation"):
        new_title = st.sidebar.text_input("Enter conversation title", value="New Conversation " + time.strftime("%H:%M:%S"))
        new_conv_id = create_new_conversation(new_title)
        st.session_state.current_conversation_id = new_conv_id
        st.session_state.chat_messages = []
        render_app()
        return

    st.title("💬 Chat with Injamul")
    model_choice = st.selectbox(
        "Select a model for responses:",
        [
          "llama-3.1-8b-instant",
          "gemma2-9b-it",
          "deepseek-r1-distill-llama-70b",
          "deepseek-r1-distill-qwen-32b",
          "qwen-2.5-32b"
        ]
    )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    input_query = st.chat_input("Type your message…")
    if input_query:
        st.session_state.chat_messages.append({"role": "user", "content": input_query, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        add_message(st.session_state.current_conversation_id, "user", input_query)
        with st.chat_message("user"):
            st.markdown(input_query)

        recent = st.session_state.chat_messages[-5:]
        previous_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent])
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(model=model_choice, api_key=GROQ_API_KEY)
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)
        result = retrieval_chain.invoke({
            "input": input_query,
            "previous_conversation": previous_context,
            "context": "\n".join([d.page_content for d in documents]),
        })
        answer = result["answer"].split("</think>")[-1].strip()
        st.session_state.chat_messages.append({"role": "assistant", "content": answer, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        add_message(st.session_state.current_conversation_id, "assistant", answer)
        with st.chat_message("assistant"):
            st.markdown(answer)
        push_to_github()

# Render the app
render_app()
