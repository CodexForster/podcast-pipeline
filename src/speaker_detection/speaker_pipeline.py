from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from .activity import ActivityEstimator
from .face_detection import FaceDetector
from .silhouette import SilhouetteRenderer
from .tracking import CentroidTracker, TrackedSpeaker
from utils.frame_enhancer import PodcastFrameEnhancer
from utils.ffmpeg_utils import mux_audio


class SpeakerDetectionPipeline:
    def __init__(self, config: dict, directories: dict, logger) -> None:
        self.config = config
        self.dirs = directories
        self.logger = logger
        self.toggle_cfg = config.get("pipeline", {}).get("speaker_detection", {})
        self.speaker_cfg = config.get("speaker_detection", {})
        self.video_cfg = config.get("video", {})
        self.outline_cfg = config.get("speaker_outline", {})
        self.production_edits_cfg = config.get("production_edits", {})
        if not self.toggle_cfg.get("podcast_visual_enhancement", True):
            self.production_edits_cfg = {**self.production_edits_cfg, "enabled": False}

        self.detector = None
        self.tracker = None
        self.activity = None
        self.enhancer = PodcastFrameEnhancer(self.production_edits_cfg, self.logger)
        self.silhouette = SilhouetteRenderer(self.outline_cfg, self.logger)
        self._build_components()

    def _build_components(self) -> None:
        if self.toggle_cfg.get("face_detection", True):
            self.detector = FaceDetector(self.speaker_cfg, self.logger)
        if self.toggle_cfg.get("face_tracking", True):
            self.tracker = CentroidTracker(
                max_disappeared=int(self.speaker_cfg.get("tracking_max_disappeared", 30)),
                max_distance=float(self.speaker_cfg.get("tracking_max_distance", 180)),
            )
        if self.toggle_cfg.get("speaker_identification", True):
            self.activity = ActivityEstimator(self.speaker_cfg, self.logger)

    def run(self, videos: List[Path]) -> List[Path]:
        if not self.toggle_cfg.get("enabled", True):
            self.logger.info("Speaker detection disabled.")
            return videos

        outputs = []
        for video in videos:
            out = self.dirs["outputs_dir"] / f"{video.stem}_annotated.mp4"
            self._process_video(video, out)
            outputs.append(out)
        return outputs

    def _process_video(self, input_video: Path, output_video: Path) -> None:
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_video = output_video.with_name(output_video.stem + ".video_only.tmp.mp4")

        writer = cv2.VideoWriter(
            str(temp_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output writer for: {temp_video}")

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                enhanced = self.enhancer.apply(frame)
                tracks = self._detect_and_track(enhanced)
                active_id = self._identify_active(enhanced, tracks)
                annotated = self._draw(enhanced, tracks, active_id)
                writer.write(annotated)
                frame_idx += 1

                if frame_idx % 150 == 0:
                    self.logger.info("Speaker detection processed %d frames for %s", frame_idx, input_video.name)
        finally:
            cap.release()
            writer.release()

        mux_audio(
            video_only_path=temp_video,
            original_media_path=input_video,
            output_path=output_video,
            video_codec=self.video_cfg.get("codec", "libx264"),
            crf=int(self.video_cfg.get("crf", 20)),
            preset=self.video_cfg.get("preset", "medium"),
        )
        if temp_video.exists():
            temp_video.unlink()
        self.logger.info("Speaker detection output written: %s", output_video)

    def _detect_and_track(self, frame) -> Dict[int, TrackedSpeaker]:
        detections = []
        if self.detector is not None:
            detections = self.detector.detect(frame)
        if self.tracker is None:
            return {}
        return self.tracker.update(detections)

    def _identify_active(self, frame, tracks: Dict[int, TrackedSpeaker]) -> int | None:
        if self.activity is None:
            return None

        lip_enabled = self.toggle_cfg.get("lip_movement_detection", True)
        motion_enabled = self.toggle_cfg.get("motion_detection", True)
        if not lip_enabled and not motion_enabled:
            # Fallback: use largest face.
            if not tracks:
                return None
            return max(tracks.items(), key=lambda kv: kv[1].bbox[2] * kv[1].bbox[3])[0]

        scores = self.activity.compute_scores(frame, tracks)
        return self.activity.choose_active_speaker(tracks, scores)

    def _draw(self, frame, tracks: Dict[int, TrackedSpeaker], active_id: int | None):
        if not self.toggle_cfg.get("draw_speaker_border", True):
            return frame
        return self.silhouette.draw_speakers(frame, tracks=tracks, active_id=active_id)

