import pandas as pd
from tqdm import tqdm
from sentence_transformers import CrossEncoder
def rerank_batch(reranker: CrossEncoder,
                 queries: list,
                 fil_query_df: pd.DataFrame,
                 top_k: int = 10,
                 batch_size: int = 50)-> pd.DataFrame:
    """
    Re-ranks retrieved documents for each query using a CrossEncoder model.

    Args:
        reranker (CrossEncoder):
            A trained CrossEncoder model (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            that takes (query, document) pairs and outputs a relevance score.

        queries (list):
            List of query strings for which re-ranking is to be performed.

        fil_query_df (pd.DataFrame):
            DataFrame containing retrieved candidate documents for each query.
            Expected columns:
              - 'query_text' : query string
              - 'query_id'   : unique query identifier
              - 'text'       : document text
              - 'corpus_id'  : document ID

        top_k (int, optional):
            Number of top-ranked documents to retain per query. Default = 10.

        batch_size (int, optional):
            Number of pairs processed in each forward pass of the CrossEncoder. Default = 50.

    Returns:
        pd.DataFrame:
            DataFrame containing top-k re-ranked results for all queries.
            Columns: ['query_id', 'corpus_id', 'rerank_score']
    """
    all_results = []

    for query in tqdm(queries, desc="Reranking queries"):
        fil_df = fil_query_df[fil_query_df.query_text==query]
        docs,doc_id = fil_df.text.tolist(),fil_df.corpus_id.tolist()
        q_id = fil_df.query_id.tolist()
        pairs = [(query, d) for d in docs]

        # ---- 2️⃣ Predict scores in batches ----
        scores = reranker.predict(pairs, batch_size=fil_df.shape[0], show_progress_bar=False)

        # ---- 4️⃣ Collect results ----
        append_df = pd.DataFrame({'query_id':q_id,'corpus_id':doc_id,'rerank_score':scores}).sort_values('rerank_score',ascending=False)
        append_df['corpus_id'] = append_df.corpus_id.apply(lambda x: x.split('_')[0])
        append_df.drop_duplicates('corpus_id',inplace=True)
        all_results.append(append_df.head(top_k))

    return pd.concat(all_results,ignore_index=True)
