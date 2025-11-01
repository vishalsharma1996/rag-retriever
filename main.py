import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_TF"] = "0"  # tell sentence-transformers NOT to use TensorFlow
import pandas as pd
import torch
import chromadb
from chromadb.config import Settings
import re
#from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer,CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import data_loader,data_preprocess,is_long_doc,actual_splitter,data_combine,generate_chunk_chroma_embeddings,model_inference,reranker,evaluate
def main():
    print('Starting main')
    # 🔹 Mapping between company names and their ticker symbols
    company_map = {
        'microsoft':'msft', 'adobe':'adbe', 'coupang':'cpng', 'linde':'lin',
        'oracle':'orcl', 'nvidia':'nvda', 'delta':'dal', 'tesla':'tsla',
        'netflix':'nflx', 'home':'hd', 'amazon':'amzn', 'apple':'aapl',
        'appl':'aapl', 'johnson':'jnj', 'jp':'jpm', 'visa':'v', 'unitedhealth':'unh',
        'google':'googl', 'alphabet':'googl', 'berkshire':'brka',
        'meta':'meta', 'pg':'pg'
    }
    reverse_company_map = {v: k for k, v in company_map.items()}

    # 🔹 Model and hardware setup
    model_name = 'mukaj/fin-mpnet-base'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 🔹 Create text splitter for long documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    # 🔹 Initialize embedding and reranker models
    model = SentenceTransformer(model_name, device=device)
    bge_reranker = CrossEncoder("BAAI/bge-reranker-large", device=device)

    # 🔹 Initialize Chroma client with persistence
    persist_dir = os.path.join('src','chroma_collection')
    os.makedirs(persist_dir,exist_ok=True)
    client = chromadb.PersistentClient(path = persist_dir)
    collection_name = "financial_docs_fin-mpnet-base"
    existing_collections = [c.name for c in client.list_collections()]

    # 🔹 Load ground-truth data and clean identifiers
    actual_data = pd.read_csv('data/FinDER_qrels.tsv', sep='\t')
    actual_data['corpus_id'] = actual_data['corpus_id'].apply(
        lambda x: re.sub('[^a-zA-Z0-9]', '', x).lower()
    )

    # 🔹 Load and preprocess data
    fin_der_data = data_loader.load_data('data/corpus.jsonl')
    query_data = data_loader.load_data('data/queries.jsonl')
    new_data_normal, query_text = data_preprocess.clean_text(
        fin_der_data, query_data, reverse_company_map
    )

    # 🔹 Identify long documents (>300 tokens)
    li_text_normal = is_long_doc.count_tokens(new_data_normal, model_name)

    # 🔹 Split those long documents semantically and recursively
    split_text_normal = actual_splitter.splitter(model, li_text_normal, splitter)

    # 🔹 Combine short and split documents into one dataset
    concated_new_data = data_combine.data_combine(new_data_normal, split_text_normal)

    # 🔹 Prepare data for Chroma embedding
    docs = [d["text"] for d in concated_new_data]
    ids = [d["_id"] for d in concated_new_data]
    metas = [{"ticker": re.split(r'(?=\d)', d["_id"])[0].lower()} for d in concated_new_data]
    if collection_name in existing_collections:
      print(f"✅ Using existing Chroma collection from {persist_dir}")
      collection = client.get_collection(collection_name)
    else:
      print(f"⚙️ Creating new Chroma collection inside {persist_dir}")
      collection = client.create_collection(name=collection_name)
      # 🔹 Generate embeddings and add to Chroma
      gen_chroma_embed = generate_chunk_chroma_embeddings.embed_and_parallelize(
          model, docs, ids, metas, collection, 500
      )
      collection = gen_chroma_embed.add_to_chroma()

    # 🔹 Retrieve top documents for each query
    fil_query_df = model_inference.get_results(
        query_data, model, actual_data,
        company_map, reverse_company_map, collection, 100
    )

    # 🔹 Rerank retrieved results using cross-encoder
    queries = fil_query_df.query_text.unique().tolist()
    new_fil_query_df = reranker.rerank_batch(bge_reranker, queries, fil_query_df, 10, 50)

    # 🔹 Evaluate retrieval performance
    metrics_df = evaluate.evaluate_retrieval(new_fil_query_df, actual_data)
    os.makedirs("results", exist_ok=True)
    output_path = "results/metrics.csv"
    metrics_df.to_csv(output_path, index=False)

if __name__ == "__main__":
  main()

