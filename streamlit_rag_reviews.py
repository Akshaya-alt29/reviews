import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime
import openai

# Load LLM API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize model and FAISS
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = None
review_chunks = []
metadata = []

st.title("🧠 Review Insights Chatbot (RAG Prototype)")

uploaded_file = st.file_uploader("Upload Ratings & Reviews Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Reviews")
    df['Review Date'] = pd.to_datetime(df['Review Date'], errors='coerce', dayfirst=True)
    
    # Extract useful text
    df['combined_text'] = df['Review Title'].fillna('') + ": " + df['Review Body'].fillna('')
    texts = df['combined_text'].tolist()
    
    # Embed and index
    embeddings = embedder.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    review_chunks = texts
    metadata = df[['Product Name', 'Review Date', 'Review Rating']].to_dict('records')
    
    st.success(f"Indexed {len(texts)} reviews. Ask your questions below.")

# Question input
query = st.text_input("Ask a question about your reviews:")

if query and index:
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k=10)

    # Get top-k relevant reviews
    context_chunks = [review_chunks[i] for i in I[0]]
    context = "\n".join(context_chunks)

    prompt = f"""
You are a helpful AI assistant. Use the context from product reviews below to answer the user question truthfully and accurately.

Context:
{context}

Question: {query}
Answer:
"""
    
    # Generate answer using OpenAI (can be swapped with Llama 2 or local model)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    st.markdown("### 🤖 Answer:")
    st.write(response['choices'][0]['message']['content'])

    with st.expander("📄 Context used for answer"):
        for chunk in context_chunks:
            st.markdown(f"- {chunk}")
