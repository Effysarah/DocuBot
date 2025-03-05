import streamlit as st
import requests
from io import BytesIO
import PyPDF2
import numpy as np
from transformers import AutoTokenizer, AutoModel
from duckduckgo_search import DDGS
import os
import datetime

# ----------------------- SQLAlchemy Setup ---------------------------
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Replace the below connection string with your actual PostgreSQL credentials
DB_URL = "postgresql+psycopg2://postgres:5656@localhost:5432/docubot"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    question = Column(Text)
    answer = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Create the table if it doesn't exist
Base.metadata.create_all(bind=engine)

def add_conversation(username: str, question: str, answer: str):
    db = SessionLocal()
    conv = Conversation(username=username, question=question, answer=answer)
    db.add(conv)
    db.commit()
    db.close()

def get_conversation_history(username: str):
    db = SessionLocal()
    history = db.query(Conversation).filter(Conversation.username == username).order_by(Conversation.timestamp).all()
    db.close()
    return history

# ----------------------- Embedding Setup ---------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

def generate_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
    norm = np.linalg.norm(embeddings)
    if norm > 0:
        embeddings = embeddings / norm
    return embeddings

# ----------------------- PDF Processing ---------------------------
def extract_pdf_text(file: BytesIO) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

def chunk_text(text: str, max_words: int = 200) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

# ----------------------- Knowledge Base (In-Memory) ---------------------------
def add_to_kb(chunks: list):
    if "kb" not in st.session_state:
        st.session_state["kb"] = []
    for chunk in chunks:
        embedding = generate_embedding(chunk)
        st.session_state["kb"].append({"text": chunk, "embedding": embedding})
    st.success(f"Added {len(chunks)} chunk(s) to the knowledge base.")

def search_kb(query: str, top_k: int = 3) -> list:
    if "kb" not in st.session_state or len(st.session_state["kb"]) == 0:
        return []
    query_emb = generate_embedding(query)
    similarities = []
    for entry in st.session_state["kb"]:
        emb = entry["embedding"]
        sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-10)
        similarities.append(sim)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_chunks = [st.session_state["kb"][i]["text"] for i in top_indices]
    return top_chunks

# ----------------------- Web Search ---------------------------
def web_search_context(query: str, max_results: int = 1) -> str:
    context = ""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        if results:
            for res in results:
                context += f"Title: {res.get('title', '')}\nSnippet: {res.get('body', '')}\nURL: {res.get('href', '')}\n\n"
    return context

# ----------------------- HuggingFace Inference API Call ---------------------------
def huggingface_chat(prompt: str, api_token: str) -> str:
    headers = {"Authorization": f"Bearer {api_token}"}
    model_url = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    payload = {"inputs": prompt}
    response = requests.post(model_url, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            return response.json()[0]['generated_text']
        except (IndexError, KeyError):
            return "Unexpected response format."
    else:
        return f"Error: {response.status_code} - {response.text}"

# ----------------------- Main Streamlit App ---------------------------
def main():
    st.set_page_config(page_title="🤖 DocuBot with KB & Web Search", layout="wide")
    st.title("🤖 DocuBot: Retrieval-Augmented Generation with KB & Web Search")
    
    # Sidebar: HuggingFace API Token and Username
    hf_api_token = st.sidebar.text_input("Enter your HuggingFace API Token 🔑", type="password")
    username = st.sidebar.text_input("Enter your username", value="guest")
    if not hf_api_token:
        st.sidebar.warning("Please enter your HuggingFace API Token to proceed.")
        st.stop()
    
    include_web = st.sidebar.checkbox("Include Web Search Context", value=False)
    uploaded_file = st.sidebar.file_uploader("📄 Upload PDF", type=["pdf"])
    if uploaded_file and st.sidebar.button("🛠️ Add PDF to KB"):
        pdf_text = extract_pdf_text(BytesIO(uploaded_file.read()))
        if pdf_text:
            chunks = chunk_text(pdf_text)
            add_to_kb(chunks)
        else:
            st.sidebar.error("PDF loaded but no text could be extracted.")
    
    # Create two tabs: Chat and Conversation History
    tab1, tab2 = st.tabs(["Chat", "Conversation History"])
    
    with tab1:
        st.subheader("Chat with the Assistant")
        question = st.text_input("💬 Ask Your Question:")
        if st.button("🔍 Get Answer"):
            if question.strip():
                kb_contexts = search_kb(question)
                kb_context = "\n\n".join(kb_contexts)
                web_context = web_search_context(question) if include_web else ""
                prompt = "Answer the question using the following contexts.\n\n"
                if kb_context:
                    prompt += f"Knowledge Base Context:\n{kb_context}\n\n"
                if web_context:
                    prompt += f"Web Search Context:\n{web_context}\n\n"
                prompt += f"Question: {question}\nAnswer:"
                with st.spinner("🤔 Processing your request..."):
                    answer = huggingface_chat(prompt, hf_api_token)
                # Save conversation in PostgreSQL
                add_conversation(username, question, answer)
                st.write("📝 **Response:**")
                st.write(answer)
            else:
                st.error("Please enter a valid question.")
    
    with tab2:
        st.subheader("Conversation History")
        history = get_conversation_history(username)
        if history:
            for entry in history:
                st.markdown(f"**Q:** {entry.question}")
                st.markdown(f"**A:** {entry.answer}")
                st.markdown(f"*Timestamp:* {entry.timestamp}")
                st.markdown("---")
        else:
            st.info("No conversation history available for this user.")

if __name__ == "__main__":
    main()
