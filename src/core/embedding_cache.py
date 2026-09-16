"""
embedding_cache.py

Frame-embedding cache to cut redundant CLIP forward passes. This is the
"efficient inference / latency" piece of the demo: the RTSP buffer holds
overlapping windows of frames (a 30s query and a 35s query five seconds
later share ~30s of frames), so without caching you re-encode the same
frames over and over.

Two tiers, both used together:
  - `_store`: exact-match cache keyed by (stream_id, timestamp), so a frame
    that was already embedded is never re-run through the model.
  - FAISS `IndexFlatIP`: an in-memory similarity index over everything ever
    cached, so you can also ask "what did we see that looked like X" across
    the whole session, not just the current buffer window — useful for a
    "has this happened before" style query.

This is intentionally in-memory (FAISS IndexFlatIP + a dict), not Redis —
appropriate for a single-process demo. Swapping the `_store` dict for a
Redis-backed one and the FAISS index for something like Redis's vector search
or a persisted FAISS index is the natural next step for a multi-process /
multi-camera deployment.
"""

import threading
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import faiss


class EmbeddingCache:
    def __init__(self, dim: int = 512, max_entries: int = 5000):
        self.dim = dim
        self.max_entries = max_entries
        self._store: Dict[str, torch.Tensor] = {}
        self._order: List[str] = []  
        self._index = faiss.IndexFlatIP(dim)
        self._index_keys: List[str] = []  
        self._lock = threading.Lock()

    @staticmethod
    def make_key(stream_id: str, timestamp: float) -> str:
        return f"{stream_id}:{round(timestamp, 2)}"

    def get(self, key: str) -> Optional[torch.Tensor]:
        with self._lock:
            return self._store.get(key)

    def put(self, key: str, embedding: torch.Tensor) -> None:
        with self._lock:
            if key in self._store:
                return
            if len(self._order) >= self.max_entries:
                oldest = self._order.pop(0)
                self._store.pop(oldest, None)

            self._store[key] = embedding.detach().cpu()
            self._order.append(key)

            vec = embedding.detach().cpu().numpy().astype("float32").reshape(1, -1)
            self._index.add(vec)
            self._index_keys.append(key)

    def search_similar(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            vec = query_embedding.detach().cpu().numpy().astype("float32").reshape(1, -1)
            scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))
            return [
                (self._index_keys[i], float(scores[0][rank]))
                for rank, i in enumerate(indices[0]) if i != -1
            ]

    def stats(self) -> dict:
        with self._lock:
            return {"cached_frames": len(self._store), "faiss_rows": self._index.ntotal}
