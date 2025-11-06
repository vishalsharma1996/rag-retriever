import mlflow
import yaml
import os
from src.set_mlflow import setup_mlflow, get_best_run, compare_data_hashes

def load_config(path='config/config.yaml'):
  """Load simple config for logging."""
  with open(path,'r') as f:
    return yaml.safe_load(f)

def validated_data_integrity(main_path="main_branch_artifacts/config.yaml",
                             current_path="config/config.yaml",
                             current_cfg=None):
  main_cfg = load_config(main_path)
  current_cfg = load_config(current_path)
  data_ok, diff = compare_data_hashes(main_cfg, current_cfg)
   # 🧠 Embed the diff in the config snapshot for record-keeping
  current_cfg["data_diff"] = diff
  current_cfg["data_integrity_passed"] = data_ok
  return current_cfg

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
