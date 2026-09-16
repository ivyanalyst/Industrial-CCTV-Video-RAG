"""
retrieval_engine.py

Real (not stubbed) visual grounding engine for the CCTV demo.

Design note on model choice
----------------------------
The reference project (starsuzi/VideoRAG, arXiv:2501.05874) computes retrieval
as a dot-product similarity search between video embeddings and text-query
embeddings in a shared space (see retrieval/inference.py in that repo — it is
literally `torch.matmul(query_features, video_features.T)` followed by a sort).
The paper's own encoder, InternVideo2, is a ~1B+ parameter model that expects
a CUDA GPU and multi-GB checkpoints, which isn't a fit for a portable
demo. This module reimplements the *same retrieval algorithm* (embed both
modalities into one space, score by cosine similarity, rank) but swaps in
OpenCLIP's ViT-B/32 as the encoder, which runs on CPU. That's the honest
scope: same retrieval strategy, lighter encoder, live frames instead of a
static benchmark corpus.

Reference:
  Jeong, S., Kim, K., Baek, J., & Hwang, S. J. (2025). VideoRAG:
  Retrieval-Augmented Generation over Video Corpus. arXiv:2501.05874.
  https://github.com/starsuzi/VideoRAG
"""

import time
import base64
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import cv2
import torch
import numpy as np
import open_clip

logger = logging.getLogger("VideoRAGRetriever")


@dataclass
class RetrievedEvidence:
    frame_index: int
    timestamp: float
    score: float
    jpeg_base64: str


@dataclass
class RetrievalResult:
    query: str
    evidence: List[RetrievedEvidence]
    reasoning: str
    total_frames_analyzed: int


class VideoRAGRetriever:
    """
    CLIP-based visual grounding engine.

    Encodes a batch of (timestamp, frame) pairs and a text query into a shared
    embedding space, scores every frame against the query by cosine similarity,
    and returns the top-k frames as timestamped, base64-encoded evidence — so
    every "alert" can be traced back to the exact moment and image that
    triggered it.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        cache: Optional["EmbeddingCache"] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained
        self.cache = cache
        self._load_model()

    def _load_model(self) -> None:
        logger.info(f"Loading {self.model_name} ({self.pretrained}) on {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.to(self.device).eval()

    # -- Encoding -----------------------------------------------------------

    @torch.no_grad()
    def encode_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        tensors = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = _to_pil(rgb)
            tensors.append(self.preprocess(pil_img))
        batch = torch.stack(tensors).to(self.device)
        feats = self.model.encode_image(batch)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_query(self, query: str) -> torch.Tensor:
        tokens = self.tokenizer([query]).to(self.device)
        feats = self.model.encode_text(tokens)
        return feats / feats.norm(dim=-1, keepdim=True)

    # -- Retrieval ------------------------------------------------------------

    def retrieve_keyframes(
        self,
        frames_with_ts: List[tuple], 
        query: str,
        top_k: int = 3,
        stream_id: str = "default",
    ) -> RetrievalResult:
        if not frames_with_ts:
            return RetrievalResult(query=query, evidence=[], reasoning="Frame buffer is empty.", total_frames_analyzed=0)

        timestamps = [t for t, _ in frames_with_ts]
        frames = [f for _, f in frames_with_ts]

        frame_feats = self._encode_with_cache(frames, timestamps, stream_id)
        query_feat = self.encode_query(query)

        sims = (frame_feats @ query_feat.T).squeeze(-1) 
        ranked = torch.argsort(sims, descending=True)

        top_k = min(top_k, len(frames))
        evidence = []
        for idx in ranked[:top_k].tolist():
            evidence.append(
                RetrievedEvidence(
                    frame_index=idx,
                    timestamp=timestamps[idx],
                    score=float(sims[idx]),
                    jpeg_base64=_frame_to_base64_jpeg(frames[idx]),
                )
            )

        reasoning = (
            f"Top match scored {evidence[0].score:.3f} cosine similarity against "
            f"'{query}' at t={evidence[0].timestamp:.2f}s."
            if evidence else "No frames scored above threshold."
        )

        return RetrievalResult(
            query=query,
            evidence=evidence,
            reasoning=reasoning,
            total_frames_analyzed=len(frames),
        )

    def _encode_with_cache(self, frames, timestamps, stream_id) -> torch.Tensor:
        if self.cache is None:
            return self.encode_frames(frames)

        feats: List[Optional[torch.Tensor]] = [None] * len(frames)
        to_encode_idx, to_encode_frames = [], []

        for i, (ts, frame) in enumerate(zip(timestamps, frames)):
            key = self.cache.make_key(stream_id, ts)
            cached = self.cache.get(key)
            if cached is not None:
                feats[i] = cached
            else:
                to_encode_idx.append(i)
                to_encode_frames.append(frame)

        if to_encode_frames:
            new_feats = self.encode_frames(to_encode_frames)
            for j, i in enumerate(to_encode_idx):
                feats[i] = new_feats[j]
                key = self.cache.make_key(stream_id, timestamps[i])
                self.cache.put(key, new_feats[j])

        return torch.stack(feats)


def _to_pil(rgb_array: np.ndarray):
    from PIL import Image
    return Image.fromarray(rgb_array)


def _frame_to_base64_jpeg(frame: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")
