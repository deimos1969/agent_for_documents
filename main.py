import os
import glob
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# --- CONFIGURATION ---
HF_MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

# SECURE FIX: Get the key from the environment, do NOT hardcode it.
HF_API_KEY = os.environ.get("HF_API_KEY")

DATA_FOLDER = "knowledge_base"

# --- GLOBAL VARIABLES ---
vectorizer = None
tfidf_matrix = None
documents = []
doc_filenames = []

def load_knowledge_base():
    """
    Loads text files and builds a light search index.
    """
    global vectorizer, tfidf_matrix, documents, doc_filenames
    
    print(f"Reading '{DATA_FOLDER}'...")
    documents = []
    doc_filenames = []
    
    # Check if folder exists
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

    # Build Search Index (TF-IDF) - Ultra fast, low RAM
    print("Building Search Index...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    print("--- ✅ KNOWLEDGE BASE READY ---")

# Load data on startup
load_knowledge_base()

# --- API HELPER ---
def query_huggingface_api(payload):
    # Safety Check
    if not HF_API_KEY:
        return {"error": "HF_API_KEY is missing. Please set it in Render Environment Variables."}

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
    return response.json()

# --- ROUTES ---
@app.get("/")
def health_check():
    return {"status": "Agent is running", "mode": "HF API (Cloud Inference)"}

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    query = request.question
    
    # Step 1: Retrieve relevant context locally
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top 2 most similar documents
    related_docs_indices = similarities.argsort()[:-3:-1]
    
    retrieved_docs = []
    sources = []
    
    for i in related_docs_indices:
        if similarities[i] > 0: # Only if it actually matches
            retrieved_docs.append(documents[i])
            sources.append(doc_filenames[i])
            
    context_text = "\n\n".join(retrieved_docs)
    
    if not retrieved_docs:
        context_text = "No specific context found."

    # Step 2: Send to Hugging Face API for the answer
    input_text = f"question: {query} context: {context_text}"
    
    print(f"Asking AI... (Context length: {len(context_text)})")
    
    try:
        api_response = query_huggingface_api({
            "inputs": input_text,
            "parameters": {
                "max_length": 150, 
                "temperature": 0.1 
            }
        })
        
        # Handle API response structure
        if isinstance(api_response, list) and "generated_text" in api_response[0]:
            answer = api_response[0]["generated_text"]
        elif isinstance(api_response, dict) and "error" in api_response:
             answer = f"System Message: {api_response['error']}"
        else:
            answer = str(api_response)

    except Exception as e:
        answer = f"Connection Error: {str(e)}"
    
    return {
        "question": query,
        "answer": answer,
        "sources": sources
    }