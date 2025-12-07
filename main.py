import os
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# --- CONFIGURATION ---
MODEL_NAME = "google/flan-t5-small"
DATA_FOLDER = "knowledge_base"

# --- GLOBAL VARIABLES ---
vectorizer = None
tfidf_matrix = None
tokenizer = None
model = None
documents = []
doc_filenames = []

def load_models_if_needed():
    global vectorizer, tfidf_matrix, tokenizer, model, documents, doc_filenames
    
    if model is not None:
        return

    print("--- 🐢 LAZY LOADING: Ultra-Lite Mode ---")
    
    try:
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        # We use Scikit-Learn instead of Neural Embeddings to save RAM
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing lib: {e}")

    # 1. Load Documents
    print(f"Reading '{DATA_FOLDER}'...")
    documents = []
    doc_filenames = []
    
    if os.path.exists(DATA_FOLDER):
        file_paths = glob.glob(os.path.join(DATA_FOLDER, "*.txt"))
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                if text.strip():
                    documents.append(text)
                    doc_filenames.append(os.path.basename(file_path))
    
    if not documents:
        documents = ["No data found."]
        doc_filenames = ["none"]

    # 2. Build Search Index (TF-IDF) - Uses almost 0 RAM
    print("Building Search Index...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    # 3. Load T5 Model (The only heavy part left)
    print("Loading T5 Model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    print("--- ✅ SYSTEM READY ---")

# --- API ---

@app.get("/")
def health_check():
    return {"status": "Agent is running", "mode": "Ultra-Lite"}

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    load_models_if_needed()
    
    # Need to import these locally to use them
    from sklearn.metrics.pairwise import cosine_similarity
    
    query = request.question
    
    # Step A: Retrieve (TF-IDF)
    # Transform query to vector
    query_vec = vectorizer.transform([query])
    # Calculate similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top 2 results
    # We use argsort to get indices of highest scores
    related_docs_indices = similarities.argsort()[:-3:-1]
    
    retrieved_docs = []
    sources = []
    
    for i in related_docs_indices:
        # Only include if similarity is > 0
        if similarities[i] > 0:
            retrieved_docs.append(documents[i])
            sources.append(doc_filenames[i])
            
    if not retrieved_docs:
        return {"question": query, "answer": "I couldn't find relevant info in the documents.", "sources": []}

    context_text = "\n\n".join(retrieved_docs)
    
    # Step B: Generate (T5)
    input_text = f"question: {query} context: {context_text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    outputs = model.generate(
        input_ids,
        max_length=150,
        num_beams=2,
        early_stopping=True
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "question": query,
        "answer": answer,
        "sources": sources
    }