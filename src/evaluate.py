import pandas as pd
def evaluate_retrieval(ret_df: pd.DataFrame,
                       ground_df: pd.DataFrame)-> pd.DataFrame:
  """
    Evaluate retrieval performance using Precision, Recall, and Mean Average Precision (MAP).

    Args:
        ret_df (pd.DataFrame):
            DataFrame containing retrieved documents for each query.
            Expected columns:
              - 'query_id'   : unique query identifier
              - 'corpus_id'  : retrieved document ID

        ground_df (pd.DataFrame):
            DataFrame containing the ground truth (relevant) documents per query.
            Expected columns:
              - 'query_id'   : unique query identifier
              - 'corpus_id'  : true relevant document ID

    Returns:
        pd.DataFrame:
            DataFrame with evaluation metrics for each query:
              - 'query_id'
              - 'precision'
              - 'recall'
              - 'map'  (Mean Average Precision)
  """
  pr_re_map_df = []
  for q_id,grp_df in ground_df.groupby('query_id'):
    ground_truth = pd.Series(grp_df.corpus_id.unique().tolist())
    ret_doc = pd.Series(ret_df[ret_df.query_id==q_id].corpus_id.unique().tolist())
    match_doc = ret_doc[ret_doc.isin(ground_truth)].tolist()
    if len(ret_doc):
      pr,re = len(match_doc)/len(ret_doc),len(match_doc)/len(ground_truth)
      ap_rel = 0
      counter = 0
      for i,doc in enumerate(ret_doc,start=1):
        if doc in ground_truth:
          counter += 1
          ap_rel += counter/i
      ap_rel = ap_rel/len(ground_truth)
      pr_re_map_df.append(pd.DataFrame({'query_id':q_id,'precision':pr,'recall':re,'map':ap_rel},index=[0]))
  return pd.concat(pr_re_map_df)
