# 🧠 RAG-Retriever  

A modular retriever pipeline for **Retrieval-Augmented Generation (RAG)** — focusing on data loading, semantic chunking, embedding generation, vector storage, reranking, and retrieval evaluation.  

GitHub: [https://github.com/vishalsharma1996/rag-retriever](https://github.com/vishalsharma1996/rag-retriever)

# 🧠 MLOps Integration

We’ve introduced MLflow-based experiment tracking to bring structure, reproducibility, and transparency to our RAG retriever experiments.
This setup enables us to compare metrics, log configurations, and automatically manage artifacts across branches.

🔧 Key Features

Branch-Aware Logging:

The main branch logs all experiment details but does not perform comparisons.

Experiment branches like mlops_integration log results and automatically compare metrics (e.g., recall) with both:

The main branch’s best run

Other runs within the same experiment branch

Automatic Configuration Logging:
Each run saves its configuration details (e.g., CUDA version, Python version, embedding model, reranker, splitter, and ChromaDB backend) inside artifacts/config_used.yaml.

Artifacts Management:
When performance improves, the best model artifacts are automatically stored in the artifacts/ directory and versioned for later reuse.

# 🚀 CI/CD Integration (GPU-Enabled)

We’ve integrated a GPU-aware CI/CD pipeline using GitHub Actions to automate experiment validation and promotion.

🔧 Key Highlights

Automated Experiment Validation: Every pull request to main automatically runs the RAG retriever pipeline on a GPU environment (CUDA 12.6 + PyTorch 2.8.0+cu126) to verify performance improvements.

Metric-Based Promotion: If recall improves, the experiment is automatically promoted — updating artifacts and merging the branch into main.

Secure Auto-Merge: Uses GitHub’s GITHUB_TOKEN for authenticated merges, ensuring only validated runs reach production.

Reproducible GPU Setup: The workflow mirrors the same CUDA and dependency setup used locally for consistent, deterministic results.

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
│   ├── mlflow_utils.py          # MLflow setup, tracking, comparison & artifact management
│   ├── config_utils.py          # Handles config loading & writing
│   ├── config/
│   │   └── config.yaml          # Base configuration (embedding, reranker, CUDA, splitter, etc.)
│   ├── artifacts/
│   │   └── config_used.yaml     # Auto-generated config snapshot per MLflow run
│   └── ...
├── requirements.txt             # Python dependencies
├── Dockerfile                   # (optional) for containerized setup
└── main.py                      # Entry point — runs retrieval + MLflow tracking

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
