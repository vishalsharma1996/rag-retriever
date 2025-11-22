import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from chromadb.api.models import Collection

def return_unique_ids(results: List[Dict[str, Any]],
                      query_id: str,
                      query_text: str)-> pd.DataFrame:
  """
    Convert ChromaDB query results into a structured pandas DataFrame.

    Args:
        results (List[Dict[str, Any]]): 
            The dictionary output returned by `collection.query()`. 
            It contains keys like 'ids', 'documents', 'metadatas', and 'distances'.

        query_id (str): 
            A unique identifier for the query (e.g., "Q1" or "query_apple_001").

        query_text (str): 
            The original text of the query that generated these results.

    Returns:
        pd.DataFrame: 
            A DataFrame containing the query info along with matched document IDs and texts.
            Columns: ['query_id', 'query_text', 'corpus_id', 'text']
  """
  id_dict = dict()
  corpus_id,doc_id = [],[]
  for ids,docs in zip(results['ids'][0],results['documents'][0]):
    corpus_id.append(ids)
    doc_id.append(docs)
  return pd.DataFrame({'query_id':[query_id]*len(corpus_id),'query_text':[query_text]*len(corpus_id),
                       'corpus_id':corpus_id,'text':doc_id})


def get_results(query_data: pd.DataFrame,
                model: SentenceTransformer,
                actual_data: pd.DataFrame,
                company_map: Dict[str, str],
                reverse_company_map: Dict[str, str],
                collection: Collection,
                n_results: int = 100)-> pd.DataFrame:
  """
    Retrieve top document results from a Chroma collection for each query using a SentenceTransformer model.

    Args:
        query_data (pd.DataFrame):
            DataFrame containing query information.
            Expected columns include '_id' (query ID) and 'text' (query string).

        model (SentenceTransformer):
            Pretrained embedding model used to encode the query text into vector representations.

        actual_data (pd.DataFrame):
            Ground truth or labeled data, used to filter only relevant queries in the final output.

        company_map (Dict[str, str]):
            Dictionary mapping company names (e.g., "Apple") to their ticker symbols (e.g., "AAPL").

        reverse_company_map (Dict[str, str]):
            Reverse mapping of ticker symbols to company names (e.g., "AAPL" → "Apple").

        collection (Collection):
            ChromaDB collection object containing document embeddings and metadata.

        n_results (int, optional):
            Number of top similar documents to retrieve per query. Default is 100.

    Returns:
        pd.DataFrame:
            A DataFrame containing retrieved results for all valid queries, 
            merged with actual data to retain only labeled queries.
  """
  n = len(query_data)
  final_df_li = []
  for ind in range(n):
    query = query_data[ind]
    query_id = query_data[ind]['_id']
    matches = [company_map[word] for word in query['text'].split(' ') if word in company_map]
    if matches:
        ticker = matches[0]
        query_embedding = model.encode([query['text']], convert_to_numpy=True)
        results = collection.query(
            query_embeddings = query_embedding,
            n_results = n_results,
            where = {'ticker':ticker}
        )
        final_df_li.append(return_unique_ids(results,query_id,query['text']))
    else:
        matches = [word for word in query['text'].split(' ') if word in reverse_company_map]
        if matches:
          ticker = matches[0]
          modified_text = query['text']
          modified_text = modified_text.replace(ticker,reverse_company_map[ticker])
          query_embedding = model.encode([modified_text], convert_to_numpy=True)
          results = collection.query(
            query_embeddings = query_embedding,
            n_results = n_results,
            where = {'ticker':ticker}
            )
          final_df_li.append(return_unique_ids(results,query_id,query['text']))
        else:
          query_embedding = model.encode(query['text'], convert_to_numpy=True)
          results = collection.query(
            query_embeddings = query_embedding,
            n_results = n_results,
            where = {'ticker':ticker}
            )
          final_df_li.append(return_unique_ids(results,query_id,query['text']))
  all_query_df = pd.concat(final_df_li)
  # Filtering for queries with labels present
  fil_query_df = all_query_df.merge(actual_data[['query_id']].drop_duplicates(),on='query_id',how='inner')
  return fil_query_df
