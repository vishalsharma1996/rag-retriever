# 🧠 RAG-Retriever  

A modular retriever pipeline for **Retrieval-Augmented Generation (RAG)** — focusing on data loading, semantic chunking, embedding generation, vector storage, reranking, and retrieval evaluation.  

GitHub: [https://github.com/vishalsharma1996/rag-retriever](https://github.com/vishalsharma1996/rag-retriever)

---
## 📂 Project Structure
rag-retriever/
├── src/
│ ├── init.py
│ ├── data_loader.py # Load and preprocess raw data
│ ├── is_long_doc.py # Identify documents exceeding token limits
│ ├── actual_splitter.py # Split long texts recursively
│ ├── data_combine.py # Combine split + short docs
│ ├── model_inference.py # Retrieve relevant documents
│ ├── evaluate.py # Evaluate retrieval performance
│ └── ... (other helper modules)
├── requirements.txt # Python dependencies
├── Dockerfile # (optional) for containerized setup
└── main.py # Entry point — orchestrates the full pipeline


---

## 🎯 Pipeline Overview

- Loads and cleans financial corpora & queries  
- Detects long documents (>300 tokens)  
- Splits them using **RecursiveCharacterTextSplitter**  
- Generates embeddings with **SentenceTransformer (`fin-mpnet-base`)**  
- Stores vector representations in **ChromaDB**  
- Retrieves top-k documents per query  
- Reranks results with **CrossEncoder (`BAAI/bge-reranker-large`)**  
- Evaluates performance (Recall, Precision, F1)  

---

## ⚙️ Quick Start (Colab / Local)pip install -r requirements.txt

### 1. Clone the repository
```bash
git clone https://github.com/vishalsharma1996/rag-retriever.git
cd rag-retriever
2. Install dependencies
pip install -r requirements.txt
3. Download NLTK data
import nltk
nltk.download('punkt_tab')
4. Run the main pipeline
python main.py
