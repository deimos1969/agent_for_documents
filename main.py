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
# ✅ UPDATED: Points to the new Router API
HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-large"

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

    # Build Search Index (TF-IDF)
    print("Building Search Index...")
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        print("--- ✅ KNOWLEDGE BASE READY ---")
    except ValueError:
        print("--- ⚠️ KNOWLEDGE BASE EMPTY (No valid words found) ---")

# Load data on startup
load_knowledge_base()

# --- API HELPER ---
def query_huggingface_api(payload):
    if not HF_API_KEY:
        return {"error": "HF_API_KEY is missing. Please set it in Render Environment Variables."}

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    try:
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
        
        # DEBUG: Print exact error if it fails
        if response.status_code != 200:
            print(f"⚠️ API Error {response.status_code}: {response.text}")

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
    return {"status": "Agent is running", "mode": "HF Router API"}

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    query = request.question
    
    # Step 1: Retrieve relevant context locally
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
        except Exception as e:
            print(f"Retrieval Error: {e}")

    # Step 2: Send to Hugging Face API
    # Note: 'flan-t5' is a text-2-text model, so we format the input as a prompt.
    input_text = f"Answer the question based on the context.\nContext: {context_text}\nQuestion: {query}"
    
    print(f"Asking AI... (Context length: {len(context_text)})")
    
    api_response = query_huggingface_api({
        "inputs": input_text,
        "parameters": {
            "max_length": 150, 
            "temperature": 0.1 
        }
    })
    
    # Step 3: Parse Answer
    answer = "Error generating answer."
    
    if isinstance(api_response, list) and len(api_response) > 0:
        if "generated_text" in api_response[0]:
            answer = api_response[0]["generated_text"]
    elif isinstance(api_response, dict):
        if "error" in api_response:
             answer = f"API Error: {api_response['error']}"
        elif "generated_text" in api_response:
            answer = api_response["generated_text"]
    
    return {
        "question": query,
        "answer": answer,
        "sources": sources
    }