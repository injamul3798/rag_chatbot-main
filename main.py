import streamlit as st
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

# Load any other .env vars you might have
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Chat with Injamul", layout="wide")

# --- Load your secret API key from Streamlit secrets ---
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

# --- Prompt template ---
prompt = ChatPromptTemplate([
    """
    You have to act like Injamul. Your bio will be given in the context. People will ask questions 
    to you and answer the questions based on the provided context only. 
    Please provide the most accurate response based on the question and answer in short.
    <context>
    {context}
    <context>
    Question: {input}
    Answer:
    """
])

# --- Streamlit UI ---
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

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Get user input
input_query = st.chat_input("Type your message…")
if input_query:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": input_query})
    with st.chat_message("user"):
        st.markdown(input_query)

    # Initialize the LLM with your secret key
    llm = ChatGroq(model=model_choice, api_key=GROQ_API_KEY)

    # Build chains
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    # Invoke and get answer
    result = retrieval_chain.invoke({"input": input_query})
    answer = result["answer"].split("</think>")[-1].strip()

    # Append & display assistant message
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
