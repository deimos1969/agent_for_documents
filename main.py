import os
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# We import these, but we won't instantiate them until later
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

# --- CONFIGURATION ---
MODEL_NAME = "google/flan-t5-small"
DATA_FOLDER = "knowledge_base"

# --- GLOBAL VARIABLES (Start as None) ---
# We do NOT load them here. We just declare them.
embedder = None
tokenizer = None
model = None
index = None
documents = []
doc_filenames = []

def load_models_if_needed():
    """
    This function checks if models are loaded. 
    If not, it loads them now. This happens on the FIRST request.
    """
    global embedder, tokenizer, model, index, documents, doc_filenames
    
    if model is not None:
        return # Already loaded, skip

    print("--- LAZY LOADING INITIATED ---")
    try:
        # 1. Load Embedder
        print("Loading Embedder...")
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # 2. Load T5 Model
        print("Loading T5 Model...")
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        
        # 3. Load Documents
        print(f"Scanning '{DATA_FOLDER}'...")
        if not os.path.exists(DATA_FOLDER):
            print(f"WARNING: Folder '{DATA_FOLDER}' not found. Using dummy data.")
            documents = ["Dummy data. Create 'knowledge_base' folder."]
            doc_filenames = ["dummy.txt"]
        else:
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

        # 4. Create Index
        print("Creating Vector Index...")
        doc_embeddings = embedder.encode(documents)
        d = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(doc_embeddings)
        
        print("--- MODELS LOADED SUCCESSFULLY ---")

    except Exception as e:
        print(f"CRITICAL ERROR LOADING MODELS: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed")

# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    # This responds INSTANTLY so Render knows the app is alive
    return {"status": "Agent is running", "models_loaded": (model is not None)}

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    # TRIGGER THE LOAD HERE
    load_models_if_needed()
    
    query = request.question
    
    # Step A: Retrieve
    query_vector = embedder.encode([query])
    k = 2
    if k > len(documents): k = len(documents)

    distances, indices = index.search(query_vector, k)
    
    retrieved_docs = []
    sources = []
    for i in indices[0]:
        if i < len(documents):
            retrieved_docs.append(documents[i])
            sources.append(doc_filenames[i])
            
    context_text = "\n\n".join(retrieved_docs)
    
    # Step B: Generate
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