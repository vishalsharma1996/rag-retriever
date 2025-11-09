import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_TF"] = "0"  # tell sentence-transformers NOT to use TensorFlow
import pandas as pd
import torch
import chromadb
import yaml
from chromadb.config import Settings
import mlflow
from datetime import datetime
import re
#from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer,CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import sys_config_info,set_mlflow,train_log
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

    # Set up mlfow
    # writing up the config file
    compare,branch = set_mlflow.setup_mlflow()
    exp = mlflow.get_experiment_by_name(name='rag-retriever-experiments')
    # Format the current time (YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sys_config_info.write_config_info(exp.experiment_id,timestamp)
    # get data and sys info
    data_info = sys_config_info.get_data_used_info()
    info_sys = sys_config_info.sys_info()
    config = train_log.load_config()
    metrics_dict = dict()
    metrics_dict['recall'] = metrics_df.recall.mean()

    # Run experiment, log results, and update artifacts if performance improves
    with mlflow.start_run(run_name=f"{branch}_run_{timestamp}") as run:
      train_log.log_mlflow_metrics(config, metrics_dict, data_info, info_sys)
      if compare:
        # validate data integrity
        set_mlflow.fetch_main_config(output_path="main_branch_artifacts/config.yaml") # fetch config file from main branch
        current_cfg = train_log.validated_data_integrity(main_path='main_branch_artifacts/config.yaml',
                                          current_path = 'config/config.yaml')
        if not current_cfg['data_integrity_passed']:
          current_cfg['experiment']['intentional_data_update'] = True
          current_cfg['experiment']['data_version'] = 'Base version changed check data diff'
        with open('config/config.yaml','w') as f:
          yaml.dump(current_cfg,f)
        # appends history of data version logs
        train_log.append_version_log(timestamp,branch,
                           main_path="main_branch_artifacts/config.yaml",
                           current_path="config/config.yaml")
        train_log.append_model_version_log(timestamp,branch,
                           main_path="main_branch_artifacts/config.yaml",
                           current_path="config/config.yaml")
        best_run_main = set_mlflow.get_best_run(metric = 'recall',branch = 'main')
        main_best_recall = best_run_main['metrics.recall'].values[0] if best_run_main is not None else 0
        if metrics_dict['recall'] > main_best_recall:
          print(" New global best! Promoting artifacts...")
          os.makedirs('artifacts',exist_ok=True)
          with open('artifacts/config_used.yaml','w') as f:
            yaml.dump(config,f)
          mlflow.log_artifacts('artifacts')
        else:
          print("🟡 No improvement over current bests.")
      else:
          print("📘 Main branch: logging only (no comparisons).")


if __name__ == "__main__":
  main()
