from app.shard_map import shard_map
from scripts.create_shards import create_shard_collections
import chromadb
BATCH_SIZE = 500
def migrate():
  '''Migrates existing chroma db collection to shards'''
  source_client = chromadb.PersistentClient(path='chroma_store')
  old = source_client.get_collection("financial_docs_fin-mpnet-base")
  # Get total documents
  total = old.count()
  print(f"Total docs in old collection: {total}")
  create_shard_collections()
  # Create shard collections
  shard_names = set(shard_map.values())
  shard_client = chromadb.PersistentClient(path='shard_store')
  shards = {name: shard_client.get_or_create_collection(name) for name in shard_names}
  for offset in range(0, total, BATCH_SIZE):
    print(f"Processing batch {offset} - {offset + BATCH_SIZE}")
    # 1. Load batch
    batch = old.get(
            ids=None,
            where=None,
            limit=BATCH_SIZE,
            offset=offset,
            include=["ids","embeddings", "documents", "metadatas"]
            )
    ids = batch["ids"]
    docs = batch["documents"]
    embs = batch["embeddings"]
    metas = batch["metadatas"]
    # 2. For each document → insert into correct shard
    bucket = {name: {"ids": [], "documents": [], "embeddings": [], "metadatas": []}
                  for name in shard_names}
    for i in range(len(ids)):
      meta = metas[i]
      company = meta.get("ticker")
      if company not in shard_map:
        print(f" Unknown company: {company}, skipping.")
        continue

      shard_name = shard_map[company]
      bucket[shard_name]["ids"].append(ids[i])
      bucket[shard_name]["documents"].append(docs[i])
      bucket[shard_name]["embeddings"].append(embs[i])
      bucket[shard_name]["metadatas"].append(metas[i])

    for name,data in bucket.items():
      if data["ids"]:
        shards[name].add(
                    ids=data["ids"],
                    documents=data["documents"],
                    embeddings=data["embeddings"],
                    metadatas=data["metadatas"]
                )
        print(f"Added {len(data['ids'])} docs → {name}")
  print("\n🎉 Sharding Migration Completed Successfully!")
