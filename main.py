import os
import glob
import requests
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# --- CORS SETTINGS (Crucial for WordPress) ---
# This allows your website to send requests to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your actual domain (e.g., ["https://my-blog.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
documents = []      # Holds the actual text chunks
doc_filenames = []  # Holds the source filename for each chunk

def load_knowledge_base():
    global vectorizer, tfidf_matrix, documents, doc_filenames
    
    print(f"--- LOADING KNOWLEDGE BASE ---")
    documents = []
    doc_filenames = []
    
    if os.path.exists(DATA_FOLDER):
        file_paths = glob.glob(os.path.join(DATA_FOLDER, "*.txt"))
        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
                    
                    # --- CHUNKING LOGIC ---
                    # Split files by double newlines so the AI gets specific sections
                    chunks = full_text.split("\n\n")
                    
                    for chunk in chunks:
                        if chunk.strip():
                            labeled_chunk = f"Source: {filename}\nContent: {chunk.strip()}"
                            documents.append(labeled_chunk)
                            doc_filenames.append(filename)
                            
                print(f" -> Loaded & Chunked: {filename}")
            except Exception as e:
                print(f" -> Error reading {file_path}: {e}")
    else:
        print(f"⚠️ CRITICAL: Folder not found at {DATA_FOLDER}")
    
    if not documents:
        print("⚠️ No documents found. Initializing empty state.")
    else:
        print(f"Total searchable chunks: {len(documents)}")
        print("Building Search Index...")
        try:
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
            return f"API Error {response.status_code}"
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "Unexpected format."
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
            
            # Get top 3 most relevant CHUNKS
            related_docs_indices = similarities.argsort()[:-4:-1]
            
            for i in related_docs_indices:
                if similarities[i] > 0:
                    context_text += documents[i] + "\n---\n"
                    if doc_filenames[i] not in sources:
                        sources.append(doc_filenames[i])
            
            # Fallback if no keywords matched (e.g., generic questions)
            if not context_text and documents:
                context_text = "\n\n".join(documents[:3])
                sources.append(doc_filenames[0])
                
        except Exception as e:
            print(f"Retrieval error: {e}")

    # --- SAFETY LIMIT ---
    # Truncate to 2500 chars to prevent API errors
    if len(context_text) > 2500:
        context_text = context_text[:2500] + "...(truncated)"

    if not context_text:
        context_text = "No documents found."

    # 2. Construct Prompt
    prompt = f"""
    You are an expert Swiss Insurance assistant. 
    Answer the question based ONLY on the context below.
    Mention the Source filename when possible.
    
    Context:
    {context_text}
    
    Question: 
    {query}
    """
    
    # 3. Call API
    answer = query_huggingface_router(prompt)
    
    return {
        "question": query,
        "answer": answer,
        "sources": sources
    }