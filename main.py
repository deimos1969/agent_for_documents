import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize FastAPI
app = FastAPI()

# --- 1. GLOBAL VARIABLES & MODEL LOADING ---
# We load these once when the app starts so we don't reload them for every request
print("Loading models... (This may take a minute)")

# USE 'flan-t5-small' FOR RENDER FREE TIER (512MB RAM LIMIT)
# If you have a paid instance (2GB+ RAM), you can change this to "google/flan-t5-base"
MODEL_NAME = "google/flan-t5-small" 

try:
    # Load Embedder
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Load Generator (T5)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Create simple in-memory vector store
    # In a real app, you would load this from a file or database
    documents = [
        "Project Apollo aims to build a reusable rocket system.",
        "The budget for Project Apollo is $50 million specifically allocated for FY2024.",
        "Sarah Connor is the Lead Engineer. She previously worked on SkyNet.",
        "The deadline for the prototype phase is December 2025."
    ]
    
    print("Indexing documents...")
    doc_embeddings = embedder.encode(documents)
    d = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(doc_embeddings)
    print("System Ready.")

except Exception as e:
    print(f"Error loading models: {e}")
    # On Render, this will show up in your logs
    raise e

# --- 2. REQUEST MODEL ---
class QueryRequest(BaseModel):
    question: str

# --- 3. API ENDPOINT ---
@app.post("/ask")
async def ask_agent(request: QueryRequest):
    query = request.question
    
    # Step A: Retrieve
    query_vector = embedder.encode([query])
    k = 2
    distances, indices = index.search(query_vector, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    context_text = "\n".join(retrieved_docs)
    
    # Step B: Generate
    input_text = f"question: {query} context: {context_text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    outputs = model.generate(
        input_ids,
        max_length=100,
        num_beams=2, # Reduced beams to save memory on free tier
        early_stopping=True
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "question": query,
        "answer": answer,
        "context_used": retrieved_docs
    }

@app.get("/")
def health_check():
    return {"status": "Agent is running"}