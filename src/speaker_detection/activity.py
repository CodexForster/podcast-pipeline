from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None

from .tracking import BBox, TrackedSpeaker


class ActivityEstimator:
    def __init__(self, config: dict, logger) -> None:
        self.logger = logger
        self.alpha = float(config.get("activity_smoothing_alpha", 0.25))
        self.use_face_mesh = bool(config.get("use_face_mesh_for_lips", True))
        self.min_activity_threshold = float(config.get("min_activity_threshold", 0.1))
        self.prev_gray = None

        self.face_mesh = None
        has_face_mesh = bool(mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"))
        if self.use_face_mesh and has_face_mesh:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        elif self.use_face_mesh and not has_face_mesh:
            self.logger.warning("MediaPipe FaceMesh unavailable. Using ROI-based lip movement fallback.")

    def compute_scores(self, frame_bgr: np.ndarray, tracks: Dict[int, TrackedSpeaker]) -> Dict[int, float]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scores: Dict[int, float] = {}

        for tid, track in tracks.items():
            lip_score = self._lip_movement_score(frame_bgr, gray, track.bbox)
            motion_score = self._motion_score(gray, track.bbox)
            raw = 0.65 * lip_score + 0.35 * motion_score
            track.smoothed_activity = (1 - self.alpha) * track.smoothed_activity + self.alpha * raw
            scores[tid] = float(track.smoothed_activity)

        self.prev_gray = gray
        return scores

    def choose_active_speaker(self, tracks: Dict[int, TrackedSpeaker], scores: Dict[int, float]) -> Optional[int]:
        if not tracks or not scores:
            return None
        best_id, best_score = max(scores.items(), key=lambda kv: kv[1])
        return best_id if best_score >= self.min_activity_threshold else None

    def _motion_score(self, gray: np.ndarray, face_bbox: BBox) -> float:
        if self.prev_gray is None:
            return 0.0
        x, y, w, h = face_bbox
        roi = self._upper_body_roi(gray, x, y, w, h)
        prev_roi = self._upper_body_roi(self.prev_gray, x, y, w, h)
        if roi.size == 0 or prev_roi.size == 0:
            return 0.0
        diff = cv2.absdiff(roi, prev_roi)
        return float(np.mean(diff) / 255.0)

    def _lip_movement_score(self, frame_bgr: np.ndarray, gray: np.ndarray, face_bbox: BBox) -> float:
        if self.prev_gray is None:
            return 0.0

        if self.face_mesh is None:
            return self._fallback_mouth_diff(gray, face_bbox)

        x, y, w, h = face_bbox
        face_crop = frame_bgr[max(0, y): y + h, max(0, x): x + w]
        if face_crop.size == 0:
            return 0.0

        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return self._fallback_mouth_diff(gray, face_bbox)

        lm = res.multi_face_landmarks[0].landmark
        mouth_ids = [13, 14, 78, 308, 82, 312]
        xs = [int(lm[i].x * w) for i in mouth_ids]
        ys = [int(lm[i].y * h) for i in mouth_ids]
        x1, x2 = max(0, min(xs) - 5), min(w, max(xs) + 5)
        y1, y2 = max(0, min(ys) - 5), min(h, max(ys) + 5)
        if x2 <= x1 or y2 <= y1:
            return self._fallback_mouth_diff(gray, face_bbox)

        curr_mouth = cv2.cvtColor(face_crop[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        prev_face = self.prev_gray[max(0, y): y + h, max(0, x): x + w]
        if prev_face.size == 0:
            return 0.0
        prev_mouth = prev_face[y1:y2, x1:x2]
        if prev_mouth.size == 0 or prev_mouth.shape != curr_mouth.shape:
            return 0.0

        diff = cv2.absdiff(curr_mouth, prev_mouth)
        return float(np.mean(diff) / 255.0)

    def _fallback_mouth_diff(self, gray: np.ndarray, face_bbox: BBox) -> float:
        x, y, w, h = face_bbox
        mouth_y1 = y + int(0.60 * h)
        mouth_y2 = y + int(0.95 * h)
        mouth_x1 = x + int(0.20 * w)
        mouth_x2 = x + int(0.80 * w)

        curr = gray[max(0, mouth_y1):mouth_y2, max(0, mouth_x1):mouth_x2]
        prev = self.prev_gray[max(0, mouth_y1):mouth_y2, max(0, mouth_x1):mouth_x2]
        if curr.size == 0 or prev.size == 0 or curr.shape != prev.shape:
            return 0.0
        return float(np.mean(cv2.absdiff(curr, prev)) / 255.0)

    @staticmethod
    def _upper_body_roi(gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        y2 = min(gray.shape[0], y + int(3.0 * h))
        x1 = max(0, x - int(0.5 * w))
        x2 = min(gray.shape[1], x + int(1.5 * w))
        return gray[max(0, y):y2, x1:x2]

