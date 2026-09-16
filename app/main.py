import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from src.core.retrieval_engine import VideoRAGRetriever
from src.core.embedding_cache import EmbeddingCache
from src.ingestion.stream_buffer import RTSPStreamBuffer

RTSP_URL = os.environ.get(
    "RTSP_URL", "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny.mp4"
)

cache = EmbeddingCache(dim=512)
retriever = VideoRAGRetriever(cache=cache)
stream_buffer = RTSPStreamBuffer(rtsp_url=RTSP_URL, max_buffer_size=300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[Lifespan] Starting RTSP ingestion for {RTSP_URL}")
    stream_buffer.start()
    yield
    print("[Lifespan] Shutting down RTSP ingestion...")
    stream_buffer.stop()


app = FastAPI(
    title="Industrial CCTV VideoRAG API",
    description=(
        "Real-time RTSP ingestion + CLIP-based visual grounding. Retrieval "
        "strategy adapted from VideoRAG (Jeong et al., 2025): frames and "
        "queries are embedded into a shared space and ranked by cosine "
        "similarity, run live against a rolling buffer instead of a static "
        "benchmark corpus."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(..., example="Did anyone walk through the unsafe area?")
    frame_count: int = Field(30, ge=1, le=300, description="Recent buffer frames to search over")
    top_k: int = Field(3, ge=1, le=10)


class Evidence(BaseModel):
    frame_index: int
    timestamp: float
    score: float
    jpeg_base64: str


class QueryResponse(BaseModel):
    query: str
    evidence: List[Evidence]
    reasoning: str
    total_frames_analyzed: int


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": retriever.device,
        "stream_info": stream_buffer.is_healthy(),
        "cache_info": cache.stats(),
    }


@app.post("/api/v1/query", response_model=QueryResponse)
def query_live_stream(request: QueryRequest):
    """
    Pulls the most recent frames from the RTSP buffer, embeds each against
    the query using CLIP, and returns the top-k matching frames as
    timestamped, base64-encoded JPEG evidence — so every answer is traceable
    to the exact moment it was grounded in.
    """
    frames_with_ts = stream_buffer.get_recent_frames(count=request.frame_count)

    if not frames_with_ts:
        raise HTTPException(
            status_code=503,
            detail="Stream buffer is empty or still connecting to the RTSP source. Retry shortly.",
        )

    result = retriever.retrieve_keyframes(
        frames_with_ts=frames_with_ts,
        query=request.query,
        top_k=request.top_k,
        stream_id=RTSP_URL,
    )

    return QueryResponse(
        query=result.query,
        evidence=[
            Evidence(
                frame_index=e.frame_index,
                timestamp=e.timestamp,
                score=e.score,
                jpeg_base64=e.jpeg_base64,
            )
            for e in result.evidence
        ],
        reasoning=result.reasoning,
        total_frames_analyzed=result.total_frames_analyzed,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
