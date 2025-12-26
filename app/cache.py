import numpy as np
import chromadb
import hashlib
import redis
import json
from chromadb.config import Settings

cache_client = chromadb.Client(
    Settings(
        persist_directory="/content/drive/MyDrive/rag-retriever/chroma_cache"
    )
)

retrieval_cache = cache_client.get_or_create_collection(
    name="semantic_retrieval_cache"
)

final_result_cache = cache_client.get_or_create_collection(
    name="final_result_cache")

SIM_THRESHOLD = 0.90

def hash_embedding(query_embedding):
  return hashlib.md5(query_embedding.tobytes()).hexdigest()

def check_retrieval_cache(query_embedding):
  res = retrieval_cache.query(query_embeddings=[query_embedding.tolist()],
        n_results=1)

  if (
        not res
        or "distances" not in res
        or not res["distances"]
        or not res["distances"][0]
    ):
        return None

  similarity = 1 - res['distances'][0][0]
  if similarity >= SIM_THRESHOLD:
    meta = res["metadatas"][0][0]
    return {'doc_ids':json.loads(meta['doc_ids'])}
  return None

def store_retrieval_cache(query_embedding, ranked_doc_ids):
    retrieval_cache.add(
        ids=[hash_embedding(query_embedding)],
        embeddings=[query_embedding.tolist()],
        metadatas=[{
            "doc_ids": json.dumps(ranked_doc_ids)
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
    try:
        key = f"rerank:{_qhash(query)}:{doc_id}"
        val = redis_client.get(key)
        return float(val) if val is not None else None
    except Exception:
        return None

def set_rerank_score(query, doc_id, score):
    try:
        key = f"rerank:{_qhash(query)}:{doc_id}"
        redis_client.set(key, score)
    except Exception:
        pass

def check_final_semantic_cache(query_embedding):
    res = final_result_cache.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=1
    )

    if (
        not res
        or "distances" not in res
        or not res["distances"]
        or not res["distances"][0]
    ):
        return None

    similarity = 1 - res["distances"][0][0]
    if similarity >= SIM_THRESHOLD:
        return json.loads(res["metadatas"][0][0]["results"])

    return None

def store_final_semantic_cache(query_embedding, results):
    final_result_cache.add(
        ids=[hash_embedding(query_embedding)],
        embeddings=[query_embedding.tolist()],
        metadatas=[{"results": json.dumps(results)}]
    )
