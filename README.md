# 🧠 RAG-Retriever  

A modular retriever pipeline for **Retrieval-Augmented Generation (RAG)** — focusing on data loading, semantic chunking, embedding generation, vector storage, reranking, and retrieval evaluation.  

GitHub: [https://github.com/vishalsharma1996/rag-retriever](https://github.com/vishalsharma1996/rag-retriever)

# 🧠 MLOps Integration
🚀 Structured Experiment Tracking with MLflow + DVC
We’ve introduced MLflow-based experiment tracking to bring structure, reproducibility, and transparency to our RAG retriever experiments.
This setup enables us to compare metrics, log configurations, and automatically manage artifacts — all while versioning data with DVC for complete lineage tracking.

🔧 Key Features
🧩 Branch-Aware Logging

The main branch logs all experiment details but does not perform comparisons.

Experiment branches (like mlops_integration) log results and automatically compare metrics (e.g., recall) against:

The main branch’s best run

Other runs within the same experiment branch

This ensures every model improvement is validated before merging.

⚙️ Automatic Configuration Logging

Each MLflow run automatically stores its full environment and setup in
artifacts/config_used.yaml, including:

CUDA and Python versions

Embedding model and reranker configuration

Splitter parameters

ChromaDB backend and collection metadata

This guarantees experiment reproducibility across environments and branches.

📦 Artifacts Management

When performance improves, the best model artifacts are automatically versioned and stored inside artifacts/ for future reuse.
Each model version is directly linked to:

Its MLflow run ID

The Git commit hash

The data version (tracked via DVC or MD5 signatures)

🧠 DVC Integration through MLflow

To ensure complete data lineage, DVC has been integrated into the MLflow pipeline.
Every MLflow run now logs:

data_files metadata, including file paths and MD5 hashes

The corresponding DVC file versions and remote storage reference

Example:
``` text
data_files:
  FinDer_qrels.tsv:
    md5: 683eec2a001505916ac63855853b5b19
    path: data/FinDER_qrels.tsv
  corpus.jsonl:
    md5: ab133181a2bea43604d05560222f4291
    path: data/corpus.jsonl
  queries.jsonl:
    md5: b662a2b042918baa84b58a2e08da2c5e
    path: data/queries.jsonl
```

These hashes are logged as MLflow parameters, giving every experiment a data fingerprint.
If a dataset changes, the hash changes — and MLflow immediately treats it as a new data version.

This creates a direct bridge between MLflow and DVC, giving full visibility into:

Which data version trained each model

When data or features changed

How performance shifted across dataset versions

🧾 Why This Matters

With MLflow + DVC, we’ve achieved:

✅ End-to-end experiment traceability

✅ Data version control tied directly to model metrics

✅ Automated artifact management and comparison across branches

Every run is now reproducible, explainable, and comparable — across time, branches, and data versions.

💬 Next Step:
Integrating CI workflows (GitHub Actions) to auto-trigger retraining when new DVC data versions or experiment configs are pushed.
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
🔁 Copy files out from the container
docker cp rag-container:/app/results/metrics.csv ./metrics.csv
docker cp rag-container:/app/src/chroma_collection ./chroma_collection
💡 Notes

Ensure NVIDIA Container Toolkit is installed — installation guide here
.

GPU version used: CUDA 12.6, compatible with torch==2.8.0+cu126.

Environment variables like TF_CPP_MIN_LOG_LEVEL and CUDA_VISIBLE_DEVICES are already handled inside main.py for cleaner logs.
