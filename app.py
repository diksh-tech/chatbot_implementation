import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ No GROQ_API_KEY found. Please set it in your .env file.")
    st.stop()

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Groq RAG Chatbot", layout="wide")

# ----------------------------
# Session State
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ----------------------------
# Hugging Face Embedding Model
# ----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Sidebar - File Upload
# ----------------------------
st.sidebar.header("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_name = uploaded_file.name
    if file_name not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.append(file_name)
        
        # Save uploaded file locally
        with open(file_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load & split PDF
        loader = PyPDFLoader(file_name)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Create vector DB & persist
        vectordb = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="chroma_db")
        vectordb.persist()
        st.sidebar.success(f"✅ {file_name} processed & added to DB")

# ----------------------------
# Sidebar - Show Uploaded PDFs
# ----------------------------
st.sidebar.markdown("### Uploaded PDFs")
for f in st.session_state.uploaded_files:
    st.sidebar.write(f"📄 {f}")

# ----------------------------
# Load Vector DB
# ----------------------------
if os.path.exists("chroma_db"):
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
else:
    retriever = None

# ----------------------------
# Main UI
# ----------------------------
st.image("https://upload.wikimedia.org/wikipedia/commons/0/04/Chatbot.jpg", width=150)
st.title("💬 PDF Q/A Chatbot with Groq & HuggingFace Embeddings")

user_query = st.text_input("Ask something about your PDFs:")

if user_query:
    if retriever:
        llm = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY, temperature=0)

        st.markdown("### 🤖 Answering your question...")

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, return_source_documents=True
        )

        result = qa_chain({"question": user_query, "chat_history": st.session_state.chat_history})

        st.session_state.chat_history.append((user_query, result["answer"]))

        st.markdown(f"**Answer:** {result['answer']}")

        st.markdown("**Sources:**")
        for doc in result["source_documents"]:
            st.markdown(f"- {doc.metadata.get('source', 'Unknown')}")
    else:
        st.warning("⚠️ Please upload a PDF to start asking questions.")

# ----------------------------
# Chat History
# ----------------------------
if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for q, a in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {a}")
