from transformers import AutoTokenizer
from typing import List, Dict, Any
def count_tokens(data: List[Dict[str, Any]],
                 model_name: str)-> List[Dict[str, Any]]:
  """
    Identifies and separates text entries exceeding 300 tokens.

    Args:
        data (List[Dict[str, Any]]): 
            List of document dictionaries containing at least a 'text' field.
        model_name (str): 
            Name of the Hugging Face model used to load the tokenizer 
            (e.g., 'bert-base-uncased').

    Returns:
        List[Dict[str, Any]]: 
            List of documents that exceed 300 tokens. 
            These are removed from the original `data` list.
  """
  excess_len = []
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  for ind,d in enumerate(data):
    if len(tokenizer.encode(d['text']))>300:
      excess_len.append(d)
      data.pop(ind)
  return excess_len
