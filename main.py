import os
import glob
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# --- CONFIGURATION ---
# NOTE: The standard public API endpoint is 'api-inference.huggingface.co'.
# If 'router.huggingface.co' was giving you trouble, this is the safer bet for raw requests.
HF_MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

# Securely get key from Render Environment
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
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    if text.strip():
                        documents.append(text)
                        doc_filenames.append(os.path.basename(file_path))
            except Exception as e:
                print(f"Skipping file {file_path}: {e}")
    
    if not documents:
        print("Warning: No documents found. Creating dummy data.")
        documents = ["No specific data available."]
        doc_filenames = ["none"]

    # Build Search Index (TF-IDF) - Ultra fast, low RAM
    print("Building Search Index...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    print("--- ✅ KNOWLEDGE BASE READY ---")

# Load data on startup
load_knowledge_base()

# --- API HELPER (FIXED) ---
def query_huggingface_api(payload):
    # Safety Check
    if not HF_API_KEY:
        return {"error": "HF_API_KEY is missing. Please set it in Render Environment Variables."}

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    try:
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
        
        # 1. DEBUGGING: Print status if it fails
        if response.status_code != 200:
            print(f"⚠️ HF API Error {response.status_code}: {response.text}")

        # 2. SAFE PARSING: Don't crash if it's not JSON
        try:
            return response.json()
        except json.JSONDecodeError:
            return {
                "error": f"API returned invalid JSON. Status: {response.status_code}", 
                "raw_content": response.text
            }
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Network Request Failed: {str(e)}"}

# --- ROUTES ---
@app.get("/")
def health_check():
    return {"status": "Agent is running", "mode": "HF API (Standard)"}

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    query = request.question
    
    # Step 1: Retrieve relevant context locally
    try:
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
            context_text = "No specific context found in documents."
            
    except Exception as e:
        print(f"Vector search error: {e}")
        context_text = "Error retrieving context."
        sources = []

    # Step 2: Send to Hugging Face API for the answer
    input_text = f"question: {query} context: {context_text}"
    
    print(f"Asking AI... (Context length: {len(context_text)})")
    
    # Call the robust API helper
    api_response = query_huggingface_api({
        "inputs": input_text,
        "parameters": {
            "max_length": 150, 
            "temperature": 0.1 
        }
    })
    
    # Step 3: Handle the answer parsing
    answer = "Error generating answer."
    
    if isinstance(api_response, list) and len(api_response) > 0 and "generated_text" in api_response[0]:
        answer = api_response[0]["generated_text"]
    elif isinstance(api_response, dict):
        if "error" in api_response:
            # Check if model is loading (Common HF issue)
            if "estimated_time" in api_response:
                answer = f"Model is loading... please try again in {int(api_response['estimated_time'])} seconds."
            else:
                answer = f"API Error: {api_response['error']} (Raw: {api_response.get('raw_content', '')})"
        elif "generated_text" in api_response:
            answer = api_response["generated_text"]
    
    return {
        "question": query,
        "answer": answer,
        "sources": sources
    }