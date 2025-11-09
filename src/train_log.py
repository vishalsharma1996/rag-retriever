import mlflow
import yaml
import os
import json
from src.set_mlflow import setup_mlflow, get_best_run, compare_data_hashes

def load_config(path='config/config.yaml'):
  """Load simple config for logging."""
  with open(path,'r') as f:
    return yaml.safe_load(f)

def validated_data_integrity(main_path="main_branch_artifacts/config.yaml",
                             current_path="config/config.yaml",
                             current_cfg=None):
  ''' Checks the data integrity'''
  main_cfg = load_config(main_path)
  current_cfg = load_config(current_path)
  data_ok, diff = compare_data_hashes(main_cfg, current_cfg)
   # 🧠 Embed the diff in the config snapshot for record-keeping
  current_cfg["data_diff"] = diff
  current_cfg["data_integrity_passed"] = data_ok
  return current_cfg

def append_version_log(timestamp,branch,
                       main_path="main_branch_artifacts/config.yaml",
                       current_path="config/config.yaml"):
  ''' Maintain data logs of all the runs in current branch if compared to main'''
  current_cfg = validated_data_integrity(main_path="main_branch_artifacts/config.yaml",
                       current_path="config/config.yaml")
  main_cfg = load_config(main_path)
  log_entry = {'timestamp':timestamp,
               'branch':branch,
               'base_main_version':main_cfg['experiment']['data_version'],
               'changed_files':[c['name'] for c in current_cfg['data_diff']['changed']],
               'new_files':[c['name'] for c in current_cfg['data_diff']['new']],
               'missing_files':current_cfg['data_diff'].get('missing',[]),
               'changed_version':[c['new_version'] for c in current_cfg['data_diff']['changed']],
               'commit':current_cfg['experiment'].get('commit','unknown')}
  path = 'logs/version_logs.json'
  os.makedirs(os.path.dirname(path),exist_ok=True)
  version_log = []
  if os.path.exists(path):
    with open(path,'r') as f:
      version_log = json.load(f)
  version_log.append(log_entry)
  with open(path,'w') as f:
    json.dump(version_log,f,indent=2)

def append_model_version_log(timestamp,branch,
                             main_path="main_branch_artifacts/config.yaml",
                             current_path='config/config.yaml'):
  current_cfg = validated_data_integrity(main_path="main_branch_artifacts/config.yaml",
                       current_path="config/config.yaml")
  main_cfg = load_config(main_path)
  log_entry = {'timestamp':timestamp,
               'branch':branch,
               'commit':current_cfg['experiment'].get('commit','unknown'),
               "embedding_model": {
               "current": current_cfg["embedding"]["model"] if current_cfg else None,
               "main": main_cfg["embedding"]["model"],
               "type": current_cfg["embedding"]["type"]},
               "reranker_model": {
                "current": current_cfg["reranker"]["model"] if current_cfg else None,
                "main": main_cfg["reranker"]["model"],
                "type": current_cfg["reranker"]["type"]},
               "vectordb": {
                "current": current_cfg["vectordb"]["backend"] if current_cfg else None,
                "main": main_cfg["vectordb"]["backend"],
                "storage": main_cfg["vectordb"]["storage"]},
               'splitter':{
                 "current_chunk_size": current_cfg["splitter"]["chunk_size"] if current_cfg else None,
                 "main_chunk_size": main_cfg["splitter"]["chunk_size"],
                 "current_chunk_overlap": current_cfg["splitter"]["chunk_overlap"] if current_cfg else None,
                 "main_chunk_overlap": main_cfg["splitter"]["chunk_overlap"],
                 "type":main_cfg['splitter']['type']}
                 }

  log_path = "logs/model_version_log.json"
  os.makedirs(os.path.dirname(log_path),exist_ok=True)
  model_version_log = []
  if os.path.exists(log_path):
    with open(log_path,'r') as f:
      model_version_log = json.load(f)
  model_version_log.append(log_entry)
  with open(log_path,'w') as f:
    json.dump(model_version_log,f,indent=2)
  

def log_mlflow_metrics(config, metrics, data_info, sys_info):
  """Logs config and metrics to mlflow."""
  for section,values in config.items():
    if isinstance(values,dict):
      for key,val in values.items():
        mlflow.log_param(f'{section}.{key}',val)
  for k,v in metrics.items():
    mlflow.log_metric(k,v)
  for k,v in data_info.items():
    mlflow.log_param(k,v)
  for k,v in sys_info.items():
    mlflow.log_param(k,v)
