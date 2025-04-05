import streamlit as st
import json, os
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

load_dotenv()

# File for persistent chat history
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def build_context(chat_history, limit=5):
    recent = chat_history[-limit:]
    context_lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_lines.append(f"{role}: {msg['content']}")
    return "\n".join(context_lines)

# Streamlit page config
st.set_page_config(page_title="Chat with Injamul", layout="wide")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Document loading & splitting
file_path = 'who_am_I.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()
full_text = "\n".join([d.page_content for d in docs])
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text(full_text)
documents = [Document(page_content=chunk) for chunk in chunks]

# Embeddings & vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embedding)
retriever = vector_store.as_retriever()

# Prompt template with previous conversation context
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

# Initialize chat history from persistent file if available
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history()

# Sidebar: Display chat history
with st.sidebar.expander("Chat History", expanded=True):
    for msg in st.session_state.chat_history:
        role = msg["role"].capitalize()
        st.markdown(f"**{role}:** {msg['content']}")

# Main UI
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

# Display conversation in main chat window
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

input_query = st.chat_input("Type your message…")
if input_query:
    # Append user message and save history
    st.session_state.chat_history.append({"role": "user", "content": input_query})
    save_history(st.session_state.chat_history)
    with st.chat_message("user"):
        st.markdown(input_query)
    
    # Build previous conversation context for the prompt
    previous_context = build_context(st.session_state.chat_history)
    
    # Initialize the LLM with your API key
    llm = ChatGroq(model=model_choice, api_key=GROQ_API_KEY)
    
    # Build chains with context included
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    
    # Invoke and get answer (pass additional context)
    result = retrieval_chain.invoke({
        "input": input_query,
        "previous_conversation": previous_context,
    })
    answer = result["answer"].split("</think>")[-1].strip()
    
    # Append assistant message and save updated history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    save_history(st.session_state.chat_history)
    with st.chat_message("assistant"):
        st.markdown(answer)
