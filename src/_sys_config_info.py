import os
import torch
import platform
def sys_info():
  info = dict()
  info['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
  info['python_version'] = platform.python_verison()
  if torch.cuda.is_available():
    info['cuda_version'] = torch.version.cuda
    info['gpu_name'] = torch.cuda.get_device_name(0)
  else:
    info['cuda_version'] = None
    info['gpu_name'] = None

  return info
def get_config_info():
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
        }
    }
