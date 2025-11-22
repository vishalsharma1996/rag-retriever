from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from chromadb.api.models import Collection
from typing import List, Dict, Any
import numpy as np
import os
class embed_and_parallelize:
  def __init__(self, model: SentenceTransformer,
               docs: List[str],
               ids: List[str],
               metas: List[Dict[str, str]],
               collection: Collection,
               batch_size: int = 500):
    """
        Initialize the embedding and parallel processing pipeline.

        Args:
            model (SentenceTransformer):
                The SentenceTransformer model used to generate vector embeddings
                from text (e.g., "all-mpnet-base-v2" or "multi-qa-MiniLM-L6-cos-v1").

            docs (List[str]):
                A list of text documents or chunks that need to be embedded.
                Each element in this list represents one document or text passage.

            ids (List[str]):
                Unique identifiers for each document. These IDs are used to store
                and retrieve documents from the ChromaDB collection.

            metas (List[Dict[str, Any]]):
                Metadata corresponding to each document, such as source, date, or topic.
                Each element is a dictionary (e.g., {"ticker": "AAPL", "source": "10-K"}).

            collection (Collection):
                A ChromaDB collection object (created via
                `client.get_or_create_collection(name="...")`) where all
                embeddings, documents, and metadata will be stored.

            batch_size (int, optional):
                Number of documents to embed or upload to ChromaDB in one batch.
                Default is 500.
    """
    self.model = model
    self.docs = docs
    self.batch_size = batch_size
    self.collection = collection
    self.ids = ids
    self.metas = metas
    self.embeddings = []
  def embed_batch(self, batch_texts: List[str]):
    """
        Embed a batch of texts using the SentenceTransformer model.

        Args:
            batch_texts (List[str]): A list of text documents/chunks to embed.

        Returns:
            np.ndarray: Numpy array of embeddings for the given batch.
    """
    return self.model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)

  def paralleize(self):
    """
        Generate embeddings for all documents in parallel using ThreadPoolExecutor.

        Returns:
            np.ndarray: Combined NumPy array of embeddings for all documents.
    """
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(self.embed_batch, self.docs[i:i + self.batch_size])
            for i in range(0, len(self.docs), self.batch_size)
        ]
        for f in tqdm(futures, desc="Generating embeddings in parallel"):
            self.embeddings.append(f.result())

    self.embeddings = np.vstack(self.embeddings)
    return self.embeddings

  def add_to_chroma(self):
    """
        Generate embeddings (if not already done) and add them to a ChromaDB collection in batches.

        Returns:
            Collection: Updated ChromaDB collection with added embeddings, documents, and metadata.
    """
    self.embeddings = self.paralleize()
    for i in tqdm(range(0, len(self.docs), self.batch_size), desc="Adding to Chroma for chunk level embeddings"):
      self.collection.add(
          ids = self.ids[i:i + self.batch_size],
          documents = self.docs[i:i + self.batch_size],
          embeddings = self.embeddings[i:i + self.batch_size],
          metadatas = self.metas[i:i + self.batch_size],
      )
    return self.collection
