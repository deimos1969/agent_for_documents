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
# ✅ NEW URL: The universal "Chat Completions" endpoint for the Router
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

# ✅ NEW MODEL: 'flan-t5' is gone. We use SmolLM2 (fast, free, supported).
# Other options: "meta-llama/Llama-3.2-3B-Instruct" (if you have access) or "mistralai/Mistral-7B-Instruct-v0.3"
HF_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

HF_API_KEY = os.environ.get("HF_API_KEY")
DATA_FOLDER = "knowledge_base"

# --- GLOBAL VARIABLES ---
vectorizer = None
tfidf_matrix = None
documents = []
doc_filenames = []

def load_knowledge_base():
    global vectorizer, tfidf_matrix, documents, doc_filenames
    print(f"Reading '{DATA_FOLDER}'...")
    documents = []
    doc_filenames = []
    
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
        documents = ["No specific data available."]
        doc_filenames = ["none"]

    print("Building Search Index...")
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        print("--- ✅ KNOWLEDGE BASE READY ---")
    except ValueError:
        print("--- ⚠️ KNOWLEDGE BASE EMPTY ---")

load_knowledge_base()

# --- API HELPER (Updated for Router) ---
def query_huggingface_router(prompt):
    if not HF_API_KEY:
        return "Error: HF_API_KEY is missing."

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    # ✅ NEW PAYLOAD FORMAT: OpenAI-compatible "messages"
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
        
        # Parse OpenAI-style response
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
    
    # 1. Retrieve Context
    context_text = "No context found."
    sources = []
    
    if vectorizer and tfidf_matrix is not None:
        try:
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            related_docs_indices = similarities.argsort()[:-3:-1]
            
            retrieved_docs = []
            for i in related_docs_indices:
                if similarities[i] > 0:
                    retrieved_docs.append(documents[i])
                    sources.append(doc_filenames[i])
            
            if retrieved_docs:
                context_text = "\n\n".join(retrieved_docs)
        except Exception:
            pass

    # 2. Construct Prompt (Chat Style)
    prompt = f"""
    You are a helpful assistant. Answer the question based ONLY on the context below.
    
    Context:
    {context_text}
    
    Question: 
    {query}
    """
    
    print(f"Asking AI ({HF_MODEL_ID})...")
    
    # 3. Call API
    answer = query_huggingface_router(prompt)
    
    return {
        "question": query,
        "answer": answer,
        "sources": sources
    }