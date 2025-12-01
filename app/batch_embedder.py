import asyncio
import torch

class EmbeddingBatcher:
    def __init__(self, embedder, batch_size=512, max_wait_ms=5):
        self.embedder = embedder
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = asyncio.Queue()

    async def enqueue(self,text):
      loop = asyncio.get_event_loop()
      future = loop.create_future()
      await self.queue.put((text,future))
      return await future

    async def worker(self):
      while True:
        batch = []
        texts = []
        item = await self.queue.get()
        batch.append(item)
        texts.append(item[0])
        try:
          while len(batch) < self.batch_size:
            item = await asyncio.wait_for(self.queue.get(),self.wait_ms/1000)
            batch.append(item)
            texts.append(item[0])
        except asyncio.TimeoutError:
          pass
        with torch.inference_mode():
          embs = self.embedder.encoder(texts,
                                       batch = self.batch_size,
                                       convert_to_tensor = True,
                                       show_progress_bar = False)

        for i, (_,future) in enumerate(batch):
          future.set_results(embs[i].cpu().numpy())
