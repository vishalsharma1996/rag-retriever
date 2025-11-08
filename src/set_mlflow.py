import mlflow
import subprocess
import hashlib
import os
import subprocess
import re

def get_git_info():
    """Return the current Git branch and commit/tag info."""
    try:
        # Get current branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode("utf-8").strip()

        # Try to get the latest tag pointing to this commit
        try:
            tag = subprocess.check_output(
                ["git", "describe", "--tags", "--exact-match"]
            ).decode("utf-8").strip()
        except subprocess.CalledProcessError:
            # If no tag, get short commit hash instead
            tag = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"]
            ).decode("utf-8").strip()

    except Exception:
        branch, tag = "unknown", "unknown"

    return {"branch": branch, "commit_or_tag": tag}


def setup_mlflow():
  """
    Sets up MLflow tracking.
    - Creates branch-specific tracking directory
    - Enables multi-level comparison for non-main branches
  """
  branch_tag_dict = get_git_info()
  branch = branch_tag_dict.get('branch')
  track_uri = f"file:./mlruns/{branch}"
  tracking = mlflow.set_tracking_uri(track_uri)
  mlflow.set_experiment('rag-retriever-experiments')
  compare = True if branch not in ['main','prod'] else False
  return compare,branch

def compute_md5(path, chunk_size=8192):
    """Return MD5 hash of a file for reproducibility checks."""
    if not os.path.exists(path):
        return None   # if file missing, just return None instead of error

    h = hashlib.md5()  # create MD5 hash object

    with open(path, "rb") as f:  # open in binary mode to read bytes
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)  # feed each chunk to the hash function

    return h.hexdigest()  # return final 32-character hex digest

def fetch_main_config(output_path="main_branch_artifacts/config.yaml"):
    """Fetch latest config_used.yaml from main branch."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        subprocess.run(
            ["git", "show", "main:config/config.yaml"],
            stdout = open(output_path, "w"),
            stderr = subprocess.PIPE,
            check = True,
        )
        print(f"✅ Fetched main/config.yaml → {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print("⚠️ Failed to fetch main branch config:", e)
        return False

def compare_data_hashes(main_cfg,current_cfg):
  """
    Compare data file hashes between main and current experiment.
    Returns True if all files match, else False.
  """
  changed, new, missing = [], [], []
  if not main_cfg or not current_cfg:
    print("⚠️ Missing configuration(s) for comparison.")
    return False
  main_data = main_cfg.get('data_files',{})
  current_data = current_cfg.get('data_files',{})
  main_version = float(re.sub('[^0-9.]*','',main_cfg.get('experiment',{}).get('data_version','v1.0')))
  all_match = True
  for name,d in current_data.items():
    main_md5 = main_data.get(name,{}).get('md5')
    current_md5 = d.get('md5')
    if main_md5 is None:
      print(f"🆕 New file detected: {name}")
      new.append({'name':name,
                  'version':'v1.0'})
      all_match = False
    elif main_md5 != current_md5:
      print(f"⚠️ Mismatch detected in: {name}")
      print(f"   main    : {main_md5}")
      print(f"   current : {current_md5}")
      new_version = f'v{main_version + 0.1}'
      changed.append({'name':name,
                      'main_md5':main_md5,
                      'current_md5':current_md5,
                      'new_version':new_version})
      all_match = False
    else:
      print(f"✅ {name} unchanged")
  for name in main_data:
    if name not in current_data:
      print(f"⚠️ Missing in current config: {name}")
      missing.append(name)
      all_match = False
  print("\n✅ Data integrity check passed!\n" if all_match else "\n❌ Data mismatch detected.\n")
  diff = {"changed": changed, "new": new, "missing": missing}
  return all_match,diff

def get_best_run(metric='recall',branch='main'):
  """
    Fetch best MLflow run by metric value from a specific branch.

  """
  track_uri = f'file:./mlruns/{branch}'
  mlflow.set_tracking_uri(track_uri)
  experiment = mlflow.get_experiment_by_name('rag-retriever-experiments')
  if not experiment:
    print(f'No experiment was found for {branch}')
    return None
  runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                            order_by=[f'metrics.{metric} DESC'])
  if not len(runs):
    print(f'No previous runs was found for {branch}')
    return None
  return runs.head(1)
