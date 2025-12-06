from sentence_transformers import SentenceTransformer,CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from chromadb import PersistentClient
import torch

def load_embedding_model(path='models/embedding',device='cpu'):
  """
    Load the SentenceTransformer embedding model from local directory.
  """
  model = SentenceTransformer(
    path,
    device=device,
    model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
                                )
  return model

def load_reranker(path="models/reranker", device='cpu'):
    """
    Load BGE CrossEncoder reranker using SentenceTransformers.
    This automatically loads tokenizer + model internally.
    """
    reranker = CrossEncoder(path, 
                            device=device,
                            model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32})
    return reranker

def load_chroma(path="chroma_store", collection_name="financial_docs_fin-mpnet-base"):
    """
    Load the persisted ChromaDB collection.
    """
    client = PersistentClient(path=path)
    collection = client.get_collection(collection_name)
    return collection

def load_all(device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load all components at API startup:
    - Embedding Model
    - Reranker Model
    - Tokenizer
    - ChromaDB Collection
    """

    print("🔥 Loading embedding model...")
    embedder = load_embedding_model(device=device)

    print("🔥 Loading reranker model...")
    reranker_model = load_reranker(device=device)

    print("🔥 Loading Chroma collection...")
    chroma_collection = load_chroma()

    print("✅ All models and DB loaded successfully!")

    return embedder, reranker_model, chroma_collection
