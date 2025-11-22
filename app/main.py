
import app.tasks
from app.celery_worker import celery_app
from app import inference
from app import loader
from config import companies
from pydantic import BaseModel
from fastapi import FastAPI,Body,HTTPException
import time
import torch
from typing import List,Dict
import ast

# ---------------------------------------------------
# 🚀 Initialize FastAPI app
# ---------------------------------------------------
app = FastAPI(
    title="Finance RAG Retriever API",
    description="Embedding, retrieval, and reranking API",
    version="1.0",
)

class QueryInput(BaseModel):
    raw: str
# ---------------------------------------------------
# 🔥 Load all models + ChromaDB collection at startup
# ---------------------------------------------------
print("🚀 Starting server... loading models...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder, reranker, chroma_collection = loader.load_all(device=device)
company_map,reverse_company_map = companies.add_company_map()
print("✅ Server is ready!")


# ---------------------------------------------------
# 📌 Health Check Endpoint
# ---------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Finance RAG Retriever FastAPI is live 🎉",
    }


# ---------------------------------------------------
# 📌 Test Endpoint — Embedding Test
# ---------------------------------------------------
@app.post("/embed")
def embed_text(body: QueryInput):
    """
    Returns embedding for a given input text.
    """
    total_start = time.time()

    # ---------------- CPU (Celery) Preprocessing ----------------
    cpu_start = time.time()
     # ---------------- Input Parsing ----------------
    raw = body.raw.strip()
    try:
        queries = ast.literal_eval(raw)

        if not isinstance(queries, list):
            raise ValueError("Input must be a Python-style list.")

        # Ensure all elements inside are strings
        if not all(isinstance(q, str) for q in queries):
            raise ValueError("All elements inside the list must be strings.")

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input format. Paste like: ['AAPL','MSFT'] Error: {str(e)}"
        )
    all_processed, all_tickers = inference.process_queries_pipeline(
        queries, company_map, reverse_company_map
    )
    cpu_time = time.time() - cpu_start

    # ---------------- GPU Batch Embedding ----------------
    # gpu_start = time.time()
    # query_embeddings = inference.embed_query(embedder, all_processed)
    # gpu_time = time.time() - gpu_start

    total_time = time.time() - total_start
    return {
        "timing": {
          "cpu_preprocessing_sec": cpu_time,
            "total_pipeline_sec": total_time
                  }
            }

    # return {
    #     "embeddings": query_embeddings.tolist(),
    #     "tickers": all_tickers,
    #     "timing": {
    #       "cpu_preprocessing_sec": cpu_time,
    #         "gpu_embedding_sec": gpu_time,
    #         "total_pipeline_sec": total_time
    #               }
    #         }


# ---------------------------------------------------
# 📌 Test Endpoint — Reranker Test (SentenceTransformer CrossEncoder)
# ---------------------------------------------------
@app.post("/rerank")
def rerank(query: str, passage: str):
    """
    Returns reranker score using CrossEncoder.predict().
    """
    score = reranker.predict([(query, passage)])[0]
    return {"score": float(score)}

# ---------------------------------------------------
# 📌 RAG Inference Endpoint — Retrieve + Rerank
# ---------------------------------------------------
@app.get("/rag")
def rag_endpoint(query: str, final_k: int = 10):
    """
    Full RAG pipeline:
    1) Embed query
    2) Retrieve with Chroma
    3) Rerank using CrossEncoder
    """
    results = inference.rag_pipeline(
        embedder=embedder,
        reranker=reranker,
        collection=chroma_collection,
        query=query,
        company_map = company_map,
        reverse_company_map = reverse_company_map,
        top_k = 100,
        final_k = final_k
    )

    return {
        "query": query,
        "top_k": final_k,
        "results": results
    }
