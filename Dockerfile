# Use NVIDIA’s official CUDA runtime base image (for GPU + PyTorch + Transformers)
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04
# Set working directory inside the container
WORKDIR /app
# Install Python, pip, and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        build-essential \
        git \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*
  # Copy requirements and install dependencies
  COPY requirements.txt .
  RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt
  # Download NLTK tokenizers (needed for sent_tokenize)
  RUN python3 -m nltk_downloader punkt_tab
  # Copy the entire project into the container
  COPY . .
  # Default command to run your main script
  CMD ['python3','main.py']
