import numpy as np
import pandas as pd
from typing import List,Dict
from sentence_transformers import SentenceTransformer
from app.celery_worker import celery_app
from app.tasks import preprocess_query_batch
from celery.result import AsyncResult

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

def embed_query(embedder, processed_queries, batch_size=32):
    """
    Generates embedding for the input query.
    """
    embeddings = embedder.encode(
        processed_queries,
        batch_size = batch_size,
        convert_to_numpy = True,
        show_progress_bar = False
    )

    return embeddings

def chroma_search(collection, query_embedding, ticker, top_k=100):
    """
    Searches the Chroma collection using the embedding.
    Returns top_k documents with metadata and distances.
    """
    results = collection.query(
          query_embeddings = query_embedding,
          n_results = top_k,
          where = {'ticker':ticker}
      )
    return results

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

def rag_pipeline(embedder, reranker, collection, queries,
                   company_map, reverse_company_map, top_k=100, final_k=10):
    """
    Complete RAG pipeline:
    1) Embed query
    2) Retrieve using Chroma
    3) Rerank using CrossEncoder
    """
    # Step 1 — Process pipeline queries
    all_processed,all_tickers = process_queries_pipeline(queries,company_map, reverse_company_map)
    query_embeddings = embed_query(embedder, all_processed)

    # Step 2 — Retriever
    chroma_results = chroma_search(collection, query_embeddings, all_tickers, top_k = top_k)

    retrieved_docs = chroma_results["ids"][0]
    retrieved_contents = chroma_results["documents"][0]

    # Step 3 — Rerank
    final_results = rerank_results(
        reranker,
        all_processed,
        retrieved_docs,
        retrieved_contents,
        final_k = final_k
    )

    return final_results
