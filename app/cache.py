import numpy as np
import chromadb
import hashlib
import redis
import json
from chromadb.config import Settings

# =====================================================
# 🔐 VERSION CONSTANTS (DEPLOYMENT-LEVEL)
# =====================================================

EMBEDDING_MODEL_V = "mukaj_fin-mpnet-base"
CHUNKING_V = "chunk_300_overlap_30"
RETRIEVAL_TOP_K = 50

RERANKER_MODEL_V = "BAAI_bge-reranker-large"
RERANK_TOP_K = 10

SIM_THRESHOLD = 0.90

# TTLs (seconds)
TTL_RERANK = 7 * 24 * 60 * 60   # 7 days

cache_client = chromadb.Client(
    Settings(
        persist_directory="/content/drive/MyDrive/rag-retriever/chroma_cache"
    )
)

# 🔹 Semantic Retrieval Cache (versioned by collection)
retrieval_cache = cache_client.get_or_create_collection(
    name=f"semantic_retrieval_cache__{EMBEDDING_MODEL_V}__{CHUNKING_V}__k{RETRIEVAL_TOP_K}"
)

# 🔹 Final Result Cache (also versioned)
final_result_cache = cache_client.get_or_create_collection(
    name=f"final_result_cache__{EMBEDDING_MODEL_V}__{CHUNKING_V}__k{RERANK_TOP_K}")

# Helpers
def hash_embedding(query_embedding):
  return hashlib.md5(query_embedding.tobytes()).hexdigest()

def _qhash(query: str):
  return hashlib.md5(query.encode("utf-8")).hexdigest()


# =====================================================
# 🧠 SEMANTIC RETRIEVAL CACHE (SKIPS VECTOR DB)
# =====================================================

def check_retrieval_cache(query_embedding):
    res = retrieval_cache.query(
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
        meta = res["metadatas"][0][0]
        return {
            "doc_ids": json.loads(meta["doc_ids"])
        }

    return None


def store_retrieval_cache(query_embedding, ranked_doc_ids):
    retrieval_cache.add(
        ids=[hash_embedding(query_embedding)],
        embeddings=[query_embedding.tolist()],
        metadatas=[{
            "doc_ids": json.dumps(ranked_doc_ids),
            "top_k": RETRIEVAL_TOP_K
        }]
    )

# =====================================================
# ⚡ RERANKER SCORE CACHE (REDIS + TTL)
# =====================================================

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=1,
    decode_responses=True
)

def get_rerank_score(query, doc_id):
    try:
        key = f"rerank:{RERANKER_MODEL_V}:{_qhash(query)}:{doc_id}"
        val = redis_client.get(key)
        return float(val) if val is not None else None
    except Exception:
        return None


def set_rerank_score(query, doc_id, score):
    try:
        key = f"rerank:{RERANKER_MODEL_V}:{_qhash(query)}:{doc_id}"
        redis_client.set(
            key,
            score,
            ex=TTL_RERANK  # ✅ TTL added here
        )
    except Exception:
        pass

# =====================================================
# 🚀 FINAL SEMANTIC CACHE (SKIPS EVERYTHING)
# =====================================================

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
        metadatas=[{
            "results": json.dumps(results),
            "rerank_top_k": RERANK_TOP_K
        }]
    )
