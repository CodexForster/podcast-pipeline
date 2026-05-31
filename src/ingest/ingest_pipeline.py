from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from utils.ffmpeg_utils import validate_media_with_ffmpeg
from utils.file_utils import copy_if_needed


@dataclass
class MediaAsset:
    path: Path
    kind: str  # video | audio | overlay | intro | outro
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class IngestResult:
    videos: List[MediaAsset]
    audio_files: List[MediaAsset]
    overlays: List[MediaAsset]
    intros: List[MediaAsset]
    outros: List[MediaAsset]


class IngestPipeline:
    def __init__(self, config: Dict, directories: Dict[str, Path], logger) -> None:
        self.config = config
        self.dirs = directories
        self.logger = logger
        self.ingest_cfg = config.get("ingest", {})
        self.toggle_cfg = config.get("pipeline", {}).get("ingest", {})

        self.video_ext = {e.lower() for e in self.ingest_cfg.get("accepted_video_ext", [])}
        self.audio_ext = {e.lower() for e in self.ingest_cfg.get("accepted_audio_ext", [])}
        self.overlay_ext = {e.lower() for e in self.ingest_cfg.get("accepted_overlay_ext", [])}
        self.strict_ffmpeg_validation = bool(self.ingest_cfg.get("strict_ffmpeg_validation", True))

    def run(
        self,
        input_path: Path,
        audio_dir: Optional[Path] = None,
        overlay_dir: Optional[Path] = None,
        intro_path: Optional[Path] = None,
        outro_path: Optional[Path] = None,
    ) -> IngestResult:
        self.logger.info("Starting ingest pipeline.")
        if not self.toggle_cfg.get("enabled", True):
            self.logger.info("Ingest step disabled. Returning empty ingest result.")
            return IngestResult([], [], [], [], [])

        videos = self._discover_media(input_path, "video")
        audio = self._discover_media(audio_dir, "audio") if audio_dir else []
        overlays = self._discover_media(overlay_dir, "overlay") if overlay_dir else []
        intros = self._collect_single(intro_path, "intro")
        outros = self._collect_single(outro_path, "outro")

        if self.toggle_cfg.get("validate_inputs", True):
            self._validate_assets(videos + audio + overlays + intros + outros)
        if self.toggle_cfg.get("extract_metadata", True):
            for asset in videos:
                asset.metadata = self._extract_video_metadata(asset.path)
        if self.toggle_cfg.get("organize_files", True):
            self._organize_assets(videos + audio + overlays + intros + outros)

        self.logger.info("Ingest complete: %d videos, %d audio, %d overlays.", len(videos), len(audio), len(overlays))
        return IngestResult(videos=videos, audio_files=audio, overlays=overlays, intros=intros, outros=outros)

    def _discover_media(self, base_path: Optional[Path], kind: str) -> List[MediaAsset]:
        if base_path is None:
            return []
        base_path = base_path.expanduser().resolve()
        if not base_path.exists():
            raise FileNotFoundError(f"{kind} input does not exist: {base_path}")

        if base_path.is_file():
            candidates = [base_path]
        else:
            candidates = [p for p in base_path.rglob("*") if p.is_file()]

        result: List[MediaAsset] = []
        for p in sorted(candidates):
            ext = p.suffix.lower()
            if kind == "video" and ext in self.video_ext:
                result.append(MediaAsset(path=p, kind="video"))
            elif kind == "audio" and ext in self.audio_ext:
                result.append(MediaAsset(path=p, kind="audio"))
            elif kind == "overlay" and ext in self.overlay_ext:
                result.append(MediaAsset(path=p, kind="overlay"))
        return result

    def _collect_single(self, path: Optional[Path], kind: str) -> List[MediaAsset]:
        if path is None:
            return []
        p = path.expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"{kind} file does not exist: {p}")
        return [MediaAsset(path=p, kind=kind)]

    def _validate_assets(self, assets: List[MediaAsset]) -> None:
        for asset in assets:
            if not asset.path.exists() or not asset.path.is_file():
                raise FileNotFoundError(f"Missing file: {asset.path}")
            if asset.kind in {"video", "audio", "intro", "outro"} and self.strict_ffmpeg_validation:
                ok = validate_media_with_ffmpeg(asset.path)
                if not ok:
                    raise RuntimeError(f"Corrupted or invalid media file detected: {asset.path}")
            self.logger.info("Validated %s asset: %s", asset.kind, asset.path)

    def _extract_video_metadata(self, video_path: Path) -> Dict[str, object]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for metadata extraction: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0
        cap.release()

        codec = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]).strip()
        metadata = {
            "fps": fps,
            "resolution": {"width": width, "height": height},
            "codec": codec,
            "duration_sec": duration,
            "frame_count": frame_count,
        }
        self.logger.info("Metadata for %s: %s", video_path.name, metadata)
        return metadata

    def _organize_assets(self, assets: List[MediaAsset]) -> None:
        raw_dir = self.dirs["raw_dir"]
        for asset in assets:
            dest = raw_dir / asset.kind / asset.path.name
            copy_if_needed(asset.path, dest)
            asset.path = dest
            self.logger.info("Organized %s -> %s", asset.kind, dest)

