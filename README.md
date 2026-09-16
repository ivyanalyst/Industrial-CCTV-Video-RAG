# Industrial CCTV VideoRAG

Real-time RTSP ingestion + explainable visual grounding for industrial CCTV, built
for scenarios like: *"an alert fired — show me the exact frame that proves it."*

## What this actually is (and isn't)

This project adapts the **retrieval strategy** from
[VideoRAG (Jeong et al., 2025)](https://arxiv.org/abs/2501.05874) — embed video
and text queries into a shared space, score by cosine similarity, rank — and
applies it to a **live rolling frame buffer** instead of a static benchmark
corpus of pre-trimmed clips.

It does **not** use the VideoRAG repo's own encoder (InternVideo2, ~1B+ params,
requires a CUDA GPU and multi-GB checkpoints). It uses OpenCLIP's `ViT-B/32`
instead, which runs on CPU and is a fair architectural substitute for a
portfolio-scale demo — same retrieval algorithm, lighter encoder. This tradeoff
is intentional and documented here rather than glossed over.

## Architecture

```
RTSP camera ─▶ RTSPStreamBuffer (background thread, ring buffer)
                     │  (timestamp, frame) tuples
                     ▼
              VideoRAGRetriever
                     │
        ┌────────────┼─────────────┐
        ▼                          ▼
  EmbeddingCache              CLIP image/text
  (FAISS IndexFlatIP,          encoders
   avoids re-encoding
   frames already seen
   in overlapping buffer
   windows)
        │
        ▼
  cosine similarity + rank
        │
        ▼
  top-k evidence: timestamp + score + base64 JPEG
```

## What each requirement maps to

| Job requirement | Implementation |
|---|---|
| RTSP/VMS ingestion | `src/ingestion/stream_buffer.py` — threaded OpenCV capture, auto-reconnect, ring buffer |
| Visual grounding / explainable alerts | `src/core/retrieval_engine.py` — every result includes the exact timestamp and JPEG frame that produced it, not just a text answer |
| Efficient inference / latency & cost | `src/core/embedding_cache.py` — exact-match cache + FAISS index so overlapping buffer windows don't re-encode the same frames |

## Setup

```bash
pip install -r requirements.txt
export RTSP_URL="rtsp://user:pass@camera-ip/stream"   # or leave unset for the demo stream
uvicorn app.main:app --reload
```

First run downloads the OpenCLIP `ViT-B/32` (`openai`) checkpoint (~350MB) —
needs internet access once; cached locally after that.

## API

`GET /health` — stream connection status, buffer depth, cache stats.

`POST /api/v1/query`
```json
{ "query": "Did anyone walk through the unsafe area?", "frame_count": 30, "top_k": 3 }
```
Returns the top-k matching frames from the last `frame_count` buffered frames,
each with `timestamp`, `score`, and `jpeg_base64` — render as
`data:image/jpeg;base64,<value>` to see the actual evidence.

## Known limitations / honest scope

- CLIP retrieval is a **single-frame** similarity match, not the temporal
  reasoning across a full clip that InternVideo2 or a video-native encoder
  would give you. Good for "does this frame show X," weaker for actions that
  unfold over several seconds (e.g. distinguishing "walking toward" vs
  "walking away from" a zone).
- No fine-tuning — this is zero-shot CLIP similarity, not a model adapted to
  this domain's alert taxonomy.
- FAISS cache uses FIFO eviction with stale-row tolerance at eviction time —
  fine at demo scale (thousands of frames), not designed for long-running
  production retention.

## Acknowledgments & Citation

The retrieval strategy in `src/core/retrieval_engine.py` (embed video and
query into a shared space, rank by cosine similarity) is adapted from the
VideoRAG paper below. This project reimplements that strategy independently
against a live RTSP buffer with a different encoder (OpenCLIP ViT-B/32
instead of the paper's InternVideo2) — no code was copied from the original
repository, only the retrieval approach.

> Jeong, S., Kim, K., Baek, J., & Hwang, S. J. (2025). *VideoRAG:
> Retrieval-Augmented Generation over Video Corpus*. arXiv:2501.05874.
> [Code](https://github.com/starsuzi/VideoRAG)

```bibtex
@article{jeong2025videorag,
  title={VideoRAG: Retrieval-Augmented Generation over Video Corpus},
  author={Jeong, Soyeong and Kim, Kangsan and Baek, Jinheon and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2501.05874},
  year={2025}
}
```

This project also uses [OpenCLIP](https://github.com/mlfoundations/open_clip)
(Ilharco et al.) as the vision-language encoder, and
[FAISS](https://github.com/facebookresearch/faiss) (Johnson et al.) for
similarity search.
