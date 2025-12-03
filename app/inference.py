import numpy as np
import pandas as pd
import torch
import os
from typing import List,Dict
from sentence_transformers import SentenceTransformer
from app.celery_worker import celery_app
from app.tasks import preprocess_query_batch
from celery.result import AsyncResult
from app.shard_map import shard_map
from scripts.migrate_to_shards import migrate
from chromadb import Client
import asyncio,chromadb
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=8)

def chunk_list(lst, size):
    """Split a list into chunks of given size."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def process_queries_pipeline(
    queries: List[str],
    company_map: Dict[str, str],
    reverse_company_map: Dict[str, str],
    chunk_size: int = 200,
    batch_size: int = 32):
    """
    Full parallel pipeline:
    - break queries into chunks
    - send chunks to Celery CPU workers
    - collect processed queries
    - batch embed everything on GPU
    """

    print("🚀 Splitting queries into chunks...")
    chunks = list(chunk_list(queries, chunk_size))

    print(f"🔥 Sending {len(chunks)} tasks to Celery CPU workers...")
    task_ids = []

    for c in chunks:
        task = preprocess_query_batch.delay(c, company_map, reverse_company_map)
        task_ids.append(task.id)

    all_processed = []
    all_tickers = []

    print("⌛ Waiting for Celery workers to finish...")

    for tid in task_ids:
        result = celery_app.AsyncResult(tid).get()
        processed, tickers = result
        all_processed.extend(processed)
        all_tickers.extend(tickers)

    print("⚡ CPU processing done")

    return all_processed, all_tickers

def embed_query(embedder, processed_queries, batch_size=512):
    with torch.inference_mode():# 🚀 faster + no grads
        embeddings = embedder.encode(
            processed_queries,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False
        )
    return embeddings.cpu().numpy()

async def async_batch_shard_search(shard_name, emb_list,
                                   top_k=100, max_batch_size=200):
    """
    Runs ONE batched search for a shard.
    All embeddings for the shard are searched together.
    """
    shard_client = chromadb.PersistentClient(path="shard_store")
    shard_collection = shard_client.get_collection(shard_name)

    loop = asyncio.get_event_loop()
    emb_list = [
        e.tolist() if hasattr(e, "tolist") else e
        for e in emb_list
    ]
    print("Shard:", shard_name, "batch_size:", len(emb_list))
    if len(emb_list) <= max_batch_size:
        # Single batch
        results = await loop.run_in_executor(
            executor,
            lambda: shard_collection.query(
                query_embeddings=emb_list, 
                n_results=top_k,
                include=["documents", "distances", "metadatas"]
            )
        )
        return results
    # Split into smaller chunks
    all_docs, all_dists, all_metas = [], [], []
    for i in range(0, len(emb_list), max_batch_size):
        chunk = emb_list[i:i + max_batch_size]
        print(f"  Processing chunk {i//max_batch_size + 1}: {len(chunk)} queries")
        chunk_results = await loop.run_in_executor(
            executor,
            lambda: shard_collection.query(
                query_embeddings=chunk, 
                n_results=top_k,
                include=["documents", "distances", "metadatas"]
            )
        )
        all_docs.extend(chunk_results["documents"])
        all_dists.extend(chunk_results["distances"])
        all_metas.extend(chunk_results["metadatas"])
    
    return {
        "documents": all_docs,
        "distances": all_dists, 
        "metadatas": all_metas
    }

async def async_chroma_search(collection, emb_list, ticker, top_k=100):
    """
    Searches the Chroma collection using the embedding.
    Returns top_k documents with metadata and distances.
    """
    loop = asyncio.get_event_loop()
    emb_list = [
        e.tolist() if hasattr(e, "tolist") else e
        for e in emb_list
    ]
    return await loop.run_in_executor(
        executor,
        lambda: collection.query(
            query_embeddings = emb_list,
            n_results=top_k,
            where={"ticker": ticker})
        )

async def async_batch_retrieve(collection, embeddings, tickers, top_k=100):
  """
    Groups all queries by shard, performs 1 batched search per shard,
    then merges results back to original order.
  """
  
  grouped = defaultdict(lambda: {"embeddings": [], "indexes": []})
  for idx, (emb, ticker) in enumerate(zip(embeddings, tickers)):
        shard_name = shard_map.get(ticker)
        if shard_name is None:
            shard_name = "fallback"
        grouped[shard_name]["embeddings"].append(emb)
        grouped[shard_name]["indexes"].append(idx)
  # -----------------------
  # 2. Create async tasks per shard
  # -----------------------
  tasks = []
  shard_order = []
  for shard_name,data in grouped.items():
    if shard_name == "fallback":
            # fallback to full DB
            tasks.append(
                async_chroma_search(collection, data["embeddings"], ticker=None, top_k=top_k)
            )
    else:
            tasks.append(
                async_batch_shard_search(
                    shard_name=shard_name,
                    emb_list=data["embeddings"],
                    top_k=top_k
                )
            )
    shard_order.append(shard_name)
  # Run all shards in parallel
  shard_results = await asyncio.gather(*tasks)
  # -----------------------
  # 3. Reconstruct results in ORIGINAL order
  # -----------------------
  final_results = [None] * len(embeddings)
  for shard_name, res in zip(shard_order, shard_results):
    indexes = grouped[shard_name]["indexes"]
    # Each embedding gets its own list of results inside batch output
    for i, original_idx in enumerate(indexes):
            final_results[original_idx] = {
                "documents": res["documents"][i],
                "distances": res["distances"][i],
                "metadatas": res["metadatas"][i]
            }
  return final_results

def rerank_results(reranker, query: str, documents: list, contents: list, final_k=10):
  """
    Reranks retrieved documents using CrossEncoder.
    documents → list of document IDs
    contents → list of text content
  """
  pairs = [(query, doc) for doc in contents]
  doc_id = [id for id in documents]
  content = [doc for doc in contents]
  scores = reranker.predict(pairs)
  result_df = pd.DataFrame({'content':content,'corpus_id':doc_id,'rerank_score':scores}).sort_values('rerank_score',ascending=False)
  result_df['corpus_id'] = result_df.corpus_id.apply(lambda x: x.split('_')[0])
  result_df.drop_duplicates('corpus_id',inplace=True)
  result_df = result_df.head(final_k)
  return [
        {
            "score": round(float(score), 2),
            "doc_id": doc_id,
            "content": content
        }
        for score, doc_id, content in zip(result_df.rerank_score.values,result_df.corpus_id.values,result_df.content.values)
        ]

# def rag_pipeline(embedder, reranker, collection, queries,
#                    company_map, reverse_company_map, top_k=100, final_k=10):
#     """
#     Complete RAG pipeline:
#     1) Embed query
#     2) Retrieve using Chroma
#     3) Rerank using CrossEncoder
#     """
#     # Step 1 — Process pipeline queries
#     all_processed,all_tickers = process_queries_pipeline(queries,company_map, reverse_company_map)
#     query_embeddings = embed_query(embedder, all_processed)

#     # Step 2 — Retriever
#     chroma_results = chroma_search(collection, query_embedding, ticker, top_k=100)

#     retrieved_docs = chroma_results["ids"][0]
#     retrieved_contents = chroma_results["documents"][0]

#     # Step 3 — Rerank
#     final_results = rerank_results(
#         reranker,
#         all_processed,
#         retrieved_docs,
#         retrieved_contents,
#         final_k = final_k
#     )

#     return final_results
