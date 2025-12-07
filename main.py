import os
import glob
import requests
import json
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# --- CONFIGURATION ---
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_API_KEY = os.environ.get("HF_API_KEY")

# ✅ ABSOLUTE PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "knowledge_base")

# --- GLOBAL VARIABLES ---
vectorizer = None
tfidf_matrix = None
documents = []
doc_filenames = []

def load_knowledge_base():
    global vectorizer, tfidf_matrix, documents, doc_filenames
    
    print(f"--- LOADING KNOWLEDGE BASE ---")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Target Data Folder: {DATA_FOLDER}")
    
    documents = []
    doc_filenames = []
    
    if os.path.exists(DATA_FOLDER):
        print(f"Files found in folder: {os.listdir(DATA_FOLDER)}")
        file_paths = glob.glob(os.path.join(DATA_FOLDER, "*.txt"))
        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    if text.strip():
                        documents.append(text)
                        doc_filenames.append(os.path.basename(file_path))
                        print(f" -> Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                print(f" -> Error reading {file_path}: {e}")
    else:
        print(f"⚠️ CRITICAL: Folder not found at {DATA_FOLDER}")
    
    if not documents:
        print("⚠️ No documents found. Initializing empty state.")
    else:
        print("Building Search Index...")
        try:
            # We keep stop_words='english' for better search on specific terms
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)
            print("--- ✅ KNOWLEDGE BASE READY ---")
        except ValueError:
            print("--- ⚠️ KNOWLEDGE BASE ERROR (Vectorizer failed) ---")

# Initialize on startup
load_knowledge_base()

# --- API HELPER ---
def query_huggingface_router(prompt):
    if not HF_API_KEY:
        return "Error: HF_API_KEY is missing."

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": HF_MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(HF_ROUTER_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"⚠️ Router Error {response.status_code}: {response.text}")
            return f"API Error {response.status_code}: {response.text}"

        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return f"Unexpected format: {result}"
            
    except Exception as e:
        return f"Connection Error: {str(e)}"

# --- ROUTES ---
@app.get("/")
def health_check():
    return {"status": "Agent is running", "model": HF_MODEL_ID}

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    query = request.question
    context_text = ""
    sources = []
    
    # 1. Retrieve Context
    if vectorizer and tfidf_matrix is not None and documents:
        try:
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            related_docs_indices = similarities.argsort()[:-3:-1]
            
            # Try to find high-quality matches first
            for i in related_docs_indices:
                if similarities[i] > 0:
                    context_text += documents[i] + "\n\n"
                    sources.append(doc_filenames[i])
            
            # --- FALLBACK MECHANISM (Solution 2) ---
            # If the search returned NOTHING (e.g., query was "Summarize this"),
            # but we actually have documents, force feed them to the AI.
            if not context_text and documents:
                print(f"⚠️ Search score was 0. Fallback: Sending all {len(documents)} docs to context.")
                context_text = "\n\n".join(documents)
                sources = doc_filenames
                
        except Exception as e:
            print(f"Retrieval error: {e}")

    # Handle case where still no context exists (e.g. no files loaded at all)
    if not context_text:
        context_text = "No documents found in knowledge base."

    # 2. Construct Prompt
    prompt = f"""
    You are a helpful assistant. Answer the question based ONLY on the context below.
    If the context is empty, say you don't know.
    
    Context:
    {context_text}
    
    Question: 
    {query}
    """
    
    print(f"Asking AI... Sources included: {sources}")
    
    # 3. Call API
    answer = query_huggingface_router(prompt)
    
    return {
        "question": query,
        "answer": answer,
        "sources": sources
    }