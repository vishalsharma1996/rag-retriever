import mlflow
import subprocess
import hashlib
import os
import subprocess

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
