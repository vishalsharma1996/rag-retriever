from chromadb import Client
import chromadb
from app.shard_map import shard_map
def create_shard_collections():
  '''Creates empty shard collection in shard_store'''
  chroma = chromadb.PersistentClient(path="shard_store")
  shard_names = set(shard_map.values())
  for name in shard_names:
    chroma.get_or_create_collection(name)
    print(f"Created shard collection: {name}")
