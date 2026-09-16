import cv2
import time
import logging
import threading
from collections import deque
from typing import List, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RTSPStreamBuffer")


class RTSPStreamBuffer:
    """
    Thread-safe circular frame buffer that continuously captures frames
    from an RTSP stream in a background thread.
    """
    def __init__(self, rtsp_url: str, max_buffer_size: int = 300, reconnect_delay: int = 5):
        """
        :param rtsp_url: RTSP camera stream URL or video file path.
        :param max_buffer_size: Max frames in memory (300 frames ~= 10s at 30 FPS).
        :param reconnect_delay: Seconds to wait before attempting RTSP reconnection.
        """
        self.rtsp_url = rtsp_url
        self.max_buffer_size = max_buffer_size
        self.reconnect_delay = reconnect_delay

        self._buffer = deque(maxlen=self.max_buffer_size)
        self._lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._is_connected = False

    def start(self) -> None:
        if self._running:
            logger.warning("Stream buffer background worker is already running.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started RTSP ingestion thread for: {self.rtsp_url}")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("Stopped RTSP ingestion thread.")

    def _capture_loop(self) -> None:
        while self._running:
            logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")
            cap = cv2.VideoCapture(self.rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                logger.error(f"Failed to open RTSP stream. Retrying in {self.reconnect_delay}s...")
                self._is_connected = False
                time.sleep(self.reconnect_delay)
                continue

            self._is_connected = True
            logger.info("Successfully connected to RTSP stream.")

            while self._running and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("Frame read empty or stream disconnected.")
                    self._is_connected = False
                    break

                timestamp = time.time()
                with self._lock:
                    self._buffer.append((timestamp, frame))

            cap.release()
            self._is_connected = False
            if self._running:
                time.sleep(self.reconnect_delay)

    def get_recent_frames(self, count: int = 30) -> List[Tuple[float, Any]]:
        """Most recent N (timestamp, frame) tuples — this is the exact shape
        VideoRAGRetriever.retrieve_keyframes expects, so no repacking needed
        between ingestion and retrieval."""
        with self._lock:
            frames = list(self._buffer)[-count:]
        return frames

    def is_healthy(self) -> dict:
        with self._lock:
            buffer_len = len(self._buffer)
        return {
            "connected": self._is_connected,
            "buffer_capacity": self.max_buffer_size,
            "current_buffer_length": buffer_len,
            "stream_url": self.rtsp_url,
        }
