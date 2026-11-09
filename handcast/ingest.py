"""Video ingestion: download, frame extraction, hand landmark detection."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class LandmarkSequence:
    """Container for a detected hand landmark sequence.

    Attributes
    ----------
    landmarks:
        ``(T, 21, 3)`` array of MediaPipe hand landmarks in normalized image
        coordinates. The 3rd axis is ``(x, y, z)`` where ``z`` is depth
        relative to the wrist.
    confidence:
        ``(T,)`` per-frame detection confidence in [0, 1]. ``NaN`` for frames
        with no detection.
    timestamps:
        ``(T,)`` frame timestamps in seconds.
    handedness:
        Per-frame handedness label ``"Left"`` or ``"Right"``, ``None`` for
        frames with no detection.
    """

    landmarks: NDArray
    confidence: NDArray
    timestamps: NDArray
    handedness: list[str | None]

    @property
    def valid_mask(self) -> NDArray:
        """Boolean mask of frames with successful detection."""
        return ~np.isnan(self.confidence)


# ------------------------------------------------------------------- downloads

def download_video(url: str, out_dir: Path | str) -> Path:
    """Download a video from a URL using ``yt-dlp``.

    Supports YouTube, Vimeo, and any other yt-dlp-compatible source. Returns
    the path to the downloaded file. For local file paths, returns the path
    unchanged.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if Path(url).exists():
        return Path(url)

    from urllib.parse import urlsplit
    if urlsplit(url).scheme != "https" or not urlsplit(url).hostname:
        raise ValueError("Expected a local video file or HTTPS URL")
    out_tmpl = str(out_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[height<=720][ext=mp4]/best[height<=720]",
        "-o",
        out_tmpl,
        "--no-playlist",
        "--max-filesize", "100M",
        "--", url,
    ]
    logger.info("Downloading video")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    if result.returncode != 0:
        raise RuntimeError("Video download failed")

    # Find the newest file in out_dir
    files = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError("yt-dlp reported success but no mp4 was produced")
    return files[-1]


# ------------------------------------------------------------------ extraction

def extract_frames(
    video_path: Path | str,
    target_fps: float = 10.0,
    max_frames: int | None = None,
) -> tuple[list[NDArray], NDArray]:
    """Extract frames at a uniform rate from a video file.

    Returns
    -------
    frames:
        List of RGB frames as ``uint8`` arrays.
    timestamps:
        ``(T,)`` float64 array of frame timestamps in seconds.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be positive and finite")
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / target_fps))

    frames: list[NDArray] = []
    timestamps: list[float] = []
    idx = 0
    pbar = tqdm(total=n_total // step, desc="extract", leave=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            timestamps.append(idx / src_fps)
            pbar.update(1)
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    pbar.close()
    cap.release()
    return frames, np.asarray(timestamps, dtype=np.float64)


# ------------------------------------------------------------------- mediapipe

def detect_hand_landmarks(
    frames: Iterable[NDArray],
    timestamps: NDArray,
    min_detection_confidence: float = 0.6,
    min_tracking_confidence: float = 0.5,
    prefer_hand: str | None = "Right",
) -> LandmarkSequence:
    """Run MediaPipe Hands across a frame sequence.

    Parameters
    ----------
    frames: iterable of RGB frames.
    timestamps: per-frame timestamps (seconds).
    min_detection_confidence, min_tracking_confidence: MediaPipe thresholds.
    prefer_hand: if two hands are visible, which one to keep. ``None`` keeps
        the highest-confidence hand.
    """
    # Import lazily so environments without mediapipe can still import this file
    import mediapipe as mp

    frames_list = list(frames)
    n = len(frames_list)
    out = np.full((n, 21, 3), np.nan, dtype=np.float64)
    conf = np.full(n, np.nan, dtype=np.float64)
    handedness: list[str | None] = [None] * n

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as detector:
        for i, frame in enumerate(tqdm(frames_list, desc="mediapipe", leave=False)):
            res = detector.process(frame)
            if not res.multi_hand_landmarks:
                continue

            # Select the preferred hand
            choices = list(zip(res.multi_hand_landmarks, res.multi_handedness))
            if prefer_hand is not None:
                filtered = [
                    (lm, h) for (lm, h) in choices if h.classification[0].label == prefer_hand
                ]
                choices = filtered or choices

            # Highest score
            lm, h = max(choices, key=lambda pair: pair[1].classification[0].score)
            out[i] = np.array([[p.x, p.y, p.z] for p in lm.landmark])
            conf[i] = float(h.classification[0].score)
            handedness[i] = h.classification[0].label

    return LandmarkSequence(
        landmarks=out,
        confidence=conf,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        handedness=handedness,
    )


def ingest(
    video_source: str | Path,
    target_fps: float = 10.0,
    cache_dir: Path | str | None = None,
) -> LandmarkSequence:
    """Convenience wrapper: download (if URL) → extract frames → detect hands."""
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
        video_path = download_video(str(video_source), tmp)
        frames, ts = extract_frames(video_path, target_fps=target_fps, max_frames=120)
    return detect_hand_landmarks(frames, ts)
