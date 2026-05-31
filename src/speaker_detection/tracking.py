from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .face_detection import FaceDetection


BBox = Tuple[int, int, int, int]


@dataclass
class TrackedSpeaker:
    speaker_id: int
    bbox: BBox
    disappeared: int = 0
    score: float = 0.0
    smoothed_activity: float = 0.0


def _center(b: BBox) -> Tuple[float, float]:
    x, y, w, h = b
    return x + w / 2.0, y + h / 2.0


class CentroidTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 180.0) -> None:
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks: Dict[int, TrackedSpeaker] = {}

    def update(self, detections: List[FaceDetection]) -> Dict[int, TrackedSpeaker]:
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid].disappeared += 1
                if self.tracks[tid].disappeared > self.max_disappeared:
                    del self.tracks[tid]
            return self.tracks

        if not self.tracks:
            for det in detections:
                self._register(det.bbox)
            return self.tracks

        track_ids = list(self.tracks.keys())
        track_centroids = np.array([_center(self.tracks[tid].bbox) for tid in track_ids], dtype=np.float32)
        det_centroids = np.array([_center(d.bbox) for d in detections], dtype=np.float32)

        dists = np.linalg.norm(track_centroids[:, None, :] - det_centroids[None, :, :], axis=2)
        rows = np.argsort(np.min(dists, axis=1))
        cols = np.argmin(dists, axis=1)[rows]

        used_rows = set()
        used_cols = set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dists[row, col] > self.max_distance:
                continue
            tid = track_ids[row]
            self.tracks[tid].bbox = detections[col].bbox
            self.tracks[tid].disappeared = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(track_ids))) - used_rows
        unused_cols = set(range(len(detections))) - used_cols

        for row in unused_rows:
            tid = track_ids[row]
            self.tracks[tid].disappeared += 1
            if self.tracks[tid].disappeared > self.max_disappeared:
                del self.tracks[tid]

        for col in unused_cols:
            self._register(detections[col].bbox)

        return self.tracks

    def _register(self, bbox: BBox) -> None:
        self.tracks[self.next_id] = TrackedSpeaker(speaker_id=self.next_id, bbox=bbox, disappeared=0)
        self.next_id += 1

