from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2

from ingest import MediaAsset
from utils.ffmpeg_utils import resolve_video_encoder, run_cmd, video_codec_args


@dataclass
class PreprocessingResult:
    processed_videos: List[Path]
    proxy_videos: List[Path]


class PreprocessingPipeline:
    def __init__(self, config: Dict, directories: Dict[str, Path], logger) -> None:
        self.config = config
        self.dirs = directories
        self.logger = logger
        self.toggle_cfg = config.get("pipeline", {}).get("preprocessing", {})
        self.video_cfg = config.get("video", {})
        self.proxy_cfg = config.get("proxy", {})
        self.frames_cfg = config.get("frame_extraction", {})
        self.audio_cfg = config.get("audio", {})
        self.video_codec = resolve_video_encoder(self.video_cfg.get("codec", "auto"), logger=self.logger)
        self.proxy_codec = resolve_video_encoder(self.proxy_cfg.get("codec", self.video_codec), logger=self.logger)

    def run(self, videos: List[MediaAsset]) -> PreprocessingResult:
        if not self.toggle_cfg.get("enabled", True):
            self.logger.info("Preprocessing disabled. Using ingest videos as-is.")
            return PreprocessingResult([v.path for v in videos], [])

        processed: List[Path] = []
        proxies: List[Path] = []

        for asset in videos:
            current = asset.path
            basename = asset.path.stem

            if self.toggle_cfg.get("normalize_framerate", True):
                out = self.dirs["processed_dir"] / f"{basename}_fps.mp4"
                self._normalize_framerate(current, out)
                current = out

            if self.toggle_cfg.get("resize_video", True):
                out = self.dirs["processed_dir"] / f"{basename}_resized.mp4"
                self._resize_video(current, out)
                current = out

            if self.toggle_cfg.get("audio_normalization", True):
                out = self.dirs["processed_dir"] / f"{basename}_audio_norm.mp4"
                self._normalize_audio(current, out)
                current = out

            processed.append(current)

            if self.toggle_cfg.get("generate_proxies", True) and self.proxy_cfg.get("enabled", True):
                proxy_path = self.dirs["proxies_dir"] / f"{basename}{self.proxy_cfg.get('suffix', '_proxy')}.mp4"
                self._generate_proxy(current, proxy_path)
                proxies.append(proxy_path)

            if self.toggle_cfg.get("extract_frames", True):
                frame_subdir = self.dirs["frames_dir"] / basename
                self._extract_frames(current, frame_subdir)

        self.logger.info("Preprocessing complete. Processed=%d, Proxies=%d", len(processed), len(proxies))
        return PreprocessingResult(processed_videos=processed, proxy_videos=proxies)

    def _normalize_framerate(self, in_path: Path, out_path: Path) -> None:
        fps = int(self.video_cfg.get("target_fps", 30))
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(in_path),
            "-r",
            str(fps),
            *video_codec_args(
                self.video_codec,
                crf=int(self.video_cfg.get("crf", 20)),
                preset=str(self.video_cfg.get("preset", "medium")),
                bitrate=self.video_cfg.get("bitrate", None),
            ),
            "-c:a",
            "copy",
            str(out_path),
        ]
        run_cmd(cmd)
        self.logger.info("Normalized FPS: %s -> %s", in_path.name, out_path.name)

    def _resize_video(self, in_path: Path, out_path: Path) -> None:
        width = int(self.video_cfg.get("output_resolution", {}).get("width", 1280))
        height = int(self.video_cfg.get("output_resolution", {}).get("height", 720))
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(in_path),
            "-vf",
            f"scale={width}:{height}",
            *video_codec_args(
                self.video_codec,
                crf=int(self.video_cfg.get("crf", 20)),
                preset=str(self.video_cfg.get("preset", "medium")),
                bitrate=self.video_cfg.get("bitrate", None),
            ),
            "-c:a",
            "copy",
            str(out_path),
        ]
        run_cmd(cmd)
        self.logger.info("Resized video: %s -> %s", in_path.name, out_path.name)

    def _generate_proxy(self, in_path: Path, out_path: Path) -> None:
        width = int(self.proxy_cfg.get("width", 640))
        height = int(self.proxy_cfg.get("height", 360))
        fps = int(self.proxy_cfg.get("fps", 15))
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(in_path),
            "-vf",
            f"scale={width}:{height}",
            "-r",
            str(fps),
            *video_codec_args(
                self.proxy_codec,
                crf=int(self.proxy_cfg.get("crf", 28)),
                preset=str(self.proxy_cfg.get("preset", "veryfast")),
                bitrate=self.proxy_cfg.get("bitrate", "2M"),
            ),
            "-an",
            str(out_path),
        ]
        run_cmd(cmd)
        self.logger.info("Generated proxy: %s", out_path.name)

    def _extract_frames(self, in_path: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        interval = max(1, int(self.frames_cfg.get("interval", 10)))
        image_format = self.frames_cfg.get("image_format", "jpg").lower()
        quality = int(self.frames_cfg.get("quality", 95))

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for frame extraction: {in_path}")

        idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                out_path = out_dir / f"frame_{idx:06d}.{image_format}"
                if image_format in {"jpg", "jpeg"}:
                    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                else:
                    cv2.imwrite(str(out_path), frame)
                saved += 1
            idx += 1
        cap.release()
        self.logger.info("Extracted %d frames from %s", saved, in_path.name)

    def _normalize_audio(self, in_path: Path, out_path: Path) -> None:
        audio_filter = self.audio_cfg.get("normalize_filter", "loudnorm=I=-16:LRA=11:TP=-1.5")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(in_path),
            "-af",
            audio_filter,
            "-c:v",
            "copy",
            "-c:a",
            self.audio_cfg.get("codec", "aac"),
            "-b:a",
            self.audio_cfg.get("bitrate", "192k"),
            str(out_path),
        ]
        run_cmd(cmd)
        self.logger.info("Normalized audio: %s -> %s", in_path.name, out_path.name)

