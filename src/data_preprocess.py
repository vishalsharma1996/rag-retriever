import re
from typing import List, Dict, Tuple, Any
def clean_text(fin_data: List[Dict[str,Any]],
               query_data: List[Dict[str,Any]],
               reverse_company_map: Dict(str,str))-> Tuple[List[Dict[str, Any]], List]:
    """
    Cleans and standardizes text fields in financial and query data.
    
    Args:
        fin_data: List of financial data entries, each as a dict.
        query_data: List of query data entries, each as a dict.
        reverse_company_map: Dictionary mapping tickers to company names (or vice versa).

    Returns:
        Cleaned list of dictionaries (e.g., processed fin_data) and query text.
    """
    new_data_normal,query_text = [],[]
    for d in fin_data:
      title = d['title'].lower()
      new_word = [[reverse_company_map[word],word] for word in title.split(' ') if word in reverse_company_map]
      if len(new_word):
        title = title.replace(new_word[0][-1],new_word[0][0])
        title =  ' '.join(dict.fromkeys(title.split()))
      text = title + ' ' + d['text']
      text = re.sub(r'[+*]',' ',text)
      text = re.sub('—','',text)
      text = re.sub('#+',' ',text)
      text = text.replace(',','').replace('•','').replace("`s",'').replace("'s",'').lower()
      text = text.replace('%',' percent ').replace('$',' dollar ')
      text = re.sub('[^a-z0-9(). ]','',text)
      text = re.sub(r'\s+',' ',text)
      # Convert (numbers) → -numbers
      text = re.sub(r'\((\d+(?:\.\d+)?)\)', r'-\1', text)
      # Remove parentheses around words
      text = re.sub(r'\(([^)0-9]+)\)', r'\1', text)
      text = text.strip()
      new_data_normal.append({'_id':d['_id'].lower(),'title':title,'text':text})
    for d in query_data:
      d['text'] = d['text'].lower().replace("`s",'').replace("'s",'')
      d['text'] = d['text'].replace('%',' percent ').replace('#',' number ').replace('$',' dollar ')
      d['text'] = re.sub('[^0-9a-z ]','',d['text'])
      d['text'] = d['text'].strip()
      query_text.append(d['text'])
    return new_data_normal,query_text
