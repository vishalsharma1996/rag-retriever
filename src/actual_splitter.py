from typing import List, Dict, Any
from src.semantic_split import semantic_split
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
def splitter(model: SentenceTransformer,
             li_text_normal: List[Dict[str,Any]],
             splitter: RecursiveCharacterTextSplitter)-> List[Dict[str, Any]]:
  """
    Splits overlength financial documents first by semantic similarity,
    then further divides them using a recursive character-based splitter.

    Args:
        li_text_normal (List[Dict[str, Any]]):
            List of financial data entries (dicts) exceeding 300 tokens.
        splitter (RecursiveCharacterTextSplitter):
            LangChain text splitter configured for 300-token chunks with 30-token overlap.

    Returns:
        List[Dict[str, Any]]:
            List of split document segments with corresponding metadata.
  """
  split_text_normal = []
  counter  = 0
  for ind in range(len(li_text_normal)):
    tokenize_li = semantic_split(li_text_normal[ind]['text'],model,similarity_threshold=0.55)
    id = li_text_normal[ind]['_id']
    title = li_text_normal[ind]['title']
    for ind2,sent in enumerate(tokenize_li):
      if len(sent)>300:
        text_li = splitter.split_text(sent)
        split_text_normal.extend([{'_id':id+'_'+str(counter)+str(ind)+str(ind2)+str(ind3),'title':title, 'text':text} for ind3,text in enumerate(text_li)])
      else:
        split_text_normal.extend([{'_id':id+'_'+str(counter)+str(ind)+str(ind2),'title':title, 'text':sent}])
      counter += 1
  return split_text_normal
