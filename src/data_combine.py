from typing import List, Dict, Tuple, Any
import re
def data_combine(new_data_normal:List[Dict[str,Any]],
                 split_text_normal:List[Dict[str,Any]])-> List[Dict[str, Any]]:
  """
    Combines unsplit documents (under 300 tokens) with newly split document chunks
    to produce a unified dataset.

    Args:
        new_data_normal (List[Dict[str, Any]]):
            List of original documents that were already under 300 tokens.
        split_text_normal (List[Dict[str, Any]]):
            List of document chunks generated from splitting overlength texts.

    Returns:
        List[Dict[str, Any]]:
            Combined list containing both original short documents and split chunks.
  """
  # removing duplicate ids from new_data_normal
  seen = {}
  for d in new_data_normal:
      _id = d['_id']
      if _id in seen:
          seen[_id] += 1
          d['_id'] = f"{_id}_{seen[_id]}"  # add suffix
      else:
          seen[_id] = 0
  concated_new_data = new_data_normal + split_text_normal
  # Precompute ticker from ID
  for d in concated_new_data:
    d["ticker"] = re.split(r'(?=\d)', d["_id"])[0].lower()
  return concated_new_data
