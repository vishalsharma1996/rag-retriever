
import app.tasks
from app.celery_worker import celery_app
from app import inference
from app import loader
from config import companies
from pydantic import BaseModel
from fastapi import FastAPI,Body,HTTPException
from app.batch_embedder import EmbeddingBatcher
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
import orjson
import asyncio
import time
import sys
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
# 📌 Makes the worker run in background
# ---------------------------------------------------
@app.on_event('startup')
async def startup_event():
  global batcher
  batcher = EmbeddingBatcher(
        embedder=embedder,
        batch_size=512,
        max_wait_ms=5
  )
  return asyncio.create_task(batcher.worker())

# ---------------------------------------------------
# 📌 Test Endpoint — For real world applications
# ---------------------------------------------------
@app.post("/embed/rt")
async def embed_realtime(body: dict):
  text = body["text"]
  embedding = await batcher.enqueue(text)
  return {"embedding": embedding.tolist()}

# ---------------------------------------------------
# 📌 Test Endpoint — For Bulk Embedding Test
# ---------------------------------------------------
@app.post("/embed")
async def embed_text(body: QueryInput):
    """
    Returns embedding for a given input text.
    """

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

    #---------------- GPU Batch Embedding ----------------
    query_embeddings = inference.embed_query(embedder, all_processed)

    # Start timer for sharded chroma retrieval
    results = await inference.async_batch_retrieve(
    collection=chroma_collection,
    embeddings=query_embeddings,
    tickers=all_tickers,
    top_k=100)
    return results



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
@app.post("/rag")
async def rag_endpoint(body: QueryInput, final_k: int = 10):
    """
    Full RAG pipeline:
    1) Embed query
    2) Retrieve with Chroma
    3) Rerank using CrossEncoder
    """
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

    start = time.perf_counter()
    results = await inference.rag_pipeline(
        embedder  = embedder,
        reranker = reranker,
        chroma_collection = chroma_collection,
        queries = queries,
        company_map = company_map,
        reverse_company_map = reverse_company_map,
        top_k = 50,
        batch_size = 1024,
        final_k = final_k
    )
    elapsed = time.perf_counter() - start
    #return {"results": results}
    return {"time_taken_seconds": round(elapsed, 4)}
