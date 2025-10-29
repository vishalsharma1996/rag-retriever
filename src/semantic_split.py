import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
def semantic_split(text: str, model: SentenceTransformer, similarity_threshold: float = 0.8) -> List[str]:
    # Sentence tokenize
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    # Get embeddings
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        # Calculate cosine similarity with previous sentence
        sim = np.dot(embeddings[i], embeddings[i - 1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i - 1]))

        # If similarity below threshold, start new chunk
        if sim < similarity_threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    # Append the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
