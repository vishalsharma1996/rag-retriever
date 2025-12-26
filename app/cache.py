import numpy as np
import chromadb
import hashlib
import redis
from chromadb.config import Settings

# ---------- Semantic Retrieval Cache ----------
cache_client = chromadb.Client(
    Settings(
        persist_directory="/content/drive/MyDrive/rag-retriever/semantic_cache"
    )
)

retrieval_cache = cache_client.get_or_create_collection(
    name="semantic_retrieval_cache"
)

SIM_THRESHOLD = 0.90

def hash_embedding(emb: np.ndarray):
    return hashlib.md5(emb.tobytes()).hexdigest()

def check_retrieval_cache(query_embedding):
  res = retrieval_cache.query(query_embeddings=[query_embedding.tolist()],
        n_results=1)

  if not res['distances']:
    return None

  similarity = 1 - res['distances'][0][0]
  if similarity >= SIM_THRESHOLD:
    return res["metadatas"][0][0]
  return None

def store_retrieval_cache(query_embedding, ranked_doc_ids):
    retrieval_cache.add(
        embeddings=[query_embedding.tolist()],
        metadatas=[{
            "doc_ids": ranked_doc_ids
        }]
    )

# ---------- Reranker Score Cache (Redis) ----------
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=1,
    decode_responses=True
)

def _qhash(query: str):
    return hashlib.md5(query.encode("utf-8")).hexdigest()

def get_rerank_score(query, doc_id):
    key = f"rerank:{_qhash(query)}:{doc_id}"
    val = redis_client.get(key)
    return float(val) if val is not None else None

def set_rerank_score(query, doc_id, score):
    key = f"rerank:{_qhash(query)}:{doc_id}"
    redis_client.set(key, score)
