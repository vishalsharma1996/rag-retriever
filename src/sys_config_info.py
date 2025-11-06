import os
import torch
import platform
import yaml
import shutil
from src import set_mlflow
def sys_info():
  """
    Collects system and hardware information useful for experiment tracking.
    Includes device type, Python version, CUDA version, and GPU name.

  """
  info = dict()
  info['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
  info['python_version'] = platform.python_version()
  if torch.cuda.is_available():
    info['cuda_version'] = torch.version.cuda
    info['gpu_name'] = torch.cuda.get_device_name(0)
  else:
    info['cuda_version'] = None
    info['gpu_name'] = None

  return info

def get_data_used_info():
  """ Gets the used data info. """
  return      {
    "corpus.jsonl": "Corpus data (contains all retrievable documents)",
    "queries.jsonl": "Query data (contains user questions or search queries)",
    "FinDER_qrels.tsv": "Ground truth mapping (links each query to its correct retrieved documents)"
                }

def get_data_files():
  data_files = {
    "FinDer_qrels.tsv": "data/FinDER_qrels.tsv",
    "corpus.jsonl": "data/corpus.jsonl",
    "queries.jsonl": "data/queries.jsonl"
                }
  return data_files

def get_config_info(exp_id,timestamp):
  """
    Returns configuration details for the retrieval pipeline,
    including the embedding model, reranker vector database setup, splitter and data info.

  """
  git_info = set_mlflow.get_git_info()
  data_files = get_data_files()
  files_info = dict()
  for name,path in data_files.items():
    files_info[name] = {'path':path,'md5':set_mlflow.compute_md5(path)}
  return {
        "embedding": {
            "model": "mukaj/fin-mpnet-base",
            "type": "SentenceTransformer"
        },
        "reranker": {
            "model": "BAAI/bge-reranker-large",
            "type": "CrossEncoder"
        },
        "vectordb": {
            "backend": "ChromaDB",
            "storage": "local"
        },
        'splitter': {
            'type':'RecursiveCharacterTextSplitter',
            'chunk_size': 300,
            'chunk_overlap': 30
        },
        'data':get_data_used_info(),
        'system':sys_info(),
        'experiment':{'name':'rag-retriever-experiments',
                      'timestamp':timestamp,
                      'experiment_id':exp_id,
                      'branch':git_info['branch'],
                      'commit':git_info['commit_or_tag'],
                      'metric_primary':'recall',
                      'intentional_data_update':False},
        'data_files':files_info
    }

def write_config_info(exp_id,timestamp,path='config/config.yaml'):
  dirpath = os.path.dirname(path)
  if os.path.exists(dirpath):
        shutil.rmtree(dirpath)   # 🧨 removes everything inside
  os.makedirs(dirpath, exist_ok=True)
  with open(path,'w') as f:
    yaml.dump(get_config_info(exp_id,timestamp),f)
