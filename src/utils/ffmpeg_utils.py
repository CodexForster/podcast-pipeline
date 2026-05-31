from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH.")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found in PATH.")


def run_cmd(cmd: Iterable[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(list(cmd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}")
    return proc


def available_video_encoders() -> set[str]:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    if proc.returncode != 0:
        return set()

    encoders: set[str] = set()
    for line in proc.stdout.splitlines():
        # Example lines:
        # " V....D libx264            H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10"
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def resolve_video_encoder(preferred: str = "auto", logger=None) -> str:
    encoders = available_video_encoders()
    if not encoders:
        fallback = "mpeg4"
        if logger:
            logger.warning("Unable to query ffmpeg encoders. Falling back to %s.", fallback)
        return fallback

    preferred = (preferred or "auto").strip().lower()
    candidates = []
    if preferred != "auto":
        candidates.append(preferred)
    candidates.extend(["libx264", "h264_videotoolbox", "mpeg4"])

    for c in candidates:
        if c in encoders:
            if logger:
                logger.info("Using FFmpeg video encoder: %s", c)
            return c

    # Last resort: first available H.264 encoder or any video encoder.
    for c in sorted(encoders):
        if "264" in c:
            if logger:
                logger.warning("Preferred encoders unavailable. Using fallback encoder: %s", c)
            return c
    c = sorted(encoders)[0]
    if logger:
        logger.warning("Preferred encoders unavailable. Using fallback encoder: %s", c)
    return c


def video_codec_args(codec: str, crf: int = 20, preset: str = "medium", bitrate: Optional[str] = None) -> list[str]:
    codec = codec.strip().lower()
    if codec == "libx264":
        return ["-c:v", codec, "-preset", preset, "-crf", str(crf)]
    if codec == "h264_videotoolbox":
        return ["-c:v", codec, "-b:v", bitrate or "5M"]
    if codec == "mpeg4":
        return ["-c:v", codec, "-q:v", "3"]
    return ["-c:v", codec]


def validate_media_with_ffmpeg(path: Path) -> bool:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0


def mux_audio(video_only_path: Path, original_media_path: Path, output_path: Path, video_codec: str, crf: int, preset: str) -> None:
    codec = resolve_video_encoder(video_codec)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_only_path),
        "-i",
        str(original_media_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        *video_codec_args(codec, crf=crf, preset=preset),
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    run_cmd(cmd)

