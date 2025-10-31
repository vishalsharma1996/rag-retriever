# 🧠 RAG-Retriever  

A modular retriever pipeline for **Retrieval-Augmented Generation (RAG)** — focusing on data loading, semantic chunking, embedding generation, vector storage, reranking, and retrieval evaluation.  

GitHub: [https://github.com/vishalsharma1996/rag-retriever](https://github.com/vishalsharma1996/rag-retriever)

---
## 📂 Project Structure

```text
rag-retriever/
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load and preprocess raw data
│   ├── is_long_doc.py           # Identify documents exceeding token limits
│   ├── actual_splitter.py       # Split long texts recursively
│   ├── data_combine.py          # Combine split + short docs
│   ├── model_inference.py       # Retrieve relevant documents
│   ├── evaluate.py              # Evaluate retrieval performance
│   └── ... (other helper modules)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # (optional) for containerized setup
└── main.py 
```

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

🐳 Run with Docker (GPU-Enabled)

After testing locally or on Colab, you can containerize and run the entire RAG pipeline in a GPU-accelerated Docker environment.
⚡ One-Line Command
docker build -t rag-retriever . && docker run --gpus all -it --name rag-container rag-retriever python3 main.py
🧠 What Happens Behind the Scenes

🏗️ Builds a Docker image named rag-retriever using the Dockerfile.

⚙️ Launches the container with GPU support via --gpus all.

🔍 Runs the full retrieval pipeline with python3 main.py.

🧩 Logs & metrics are visible directly in your terminal

🧰 Helpful Commands

▶️ Restart without rebuilding
docker start -ai rag-container
🔁 Rebuild fresh
docker rm -f rag-container && docker rmi rag-retriever

💡 Notes

Ensure NVIDIA Container Toolkit is installed — installation guide here
.

GPU version used: CUDA 12.6, compatible with torch==2.8.0+cu126.

Environment variables like TF_CPP_MIN_LOG_LEVEL and CUDA_VISIBLE_DEVICES are already handled inside main.py for cleaner logs.
