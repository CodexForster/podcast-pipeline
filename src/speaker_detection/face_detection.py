from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


BBox = Tuple[int, int, int, int]


@dataclass
class FaceDetection:
    bbox: BBox
    confidence: float


class FaceDetector:
    def __init__(self, config: dict, logger) -> None:
        self.config = config
        self.logger = logger
        self.backend = config.get("detector_backend", "mediapipe")
        self.min_conf = float(config.get("detection_confidence", 0.5))
        self.min_face_size = int(config.get("min_face_size", 32))

        self.face_detection = None
        self.dnn_net = None
        self.haar = None
        self._has_mp_solutions = bool(
            mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection")
        )

        if self.backend == "mediapipe":
            if not self._has_mp_solutions:
                self.logger.warning(
                    "MediaPipe installed without solutions API. Falling back to Haar cascade detector."
                )
                self._init_haar()
                self.backend = "haar"
            else:
                self.face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=self.min_conf,
                )
        elif self.backend == "opencv_dnn":
            prototxt = config.get("dnn_prototxt", "")
            model = config.get("dnn_model", "")
            if prototxt and model and Path(prototxt).exists() and Path(model).exists():
                self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt, model)
            else:
                self.logger.warning("DNN model files are missing. Falling back to Haar cascade detector.")
                self._init_haar()
                self.backend = "haar"
        else:
            raise ValueError(f"Unsupported face detector backend: {self.backend}")

        self.logger.info("Face detector initialized: %s", self.backend)

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        if self.backend == "mediapipe":
            return self._detect_mediapipe(frame_bgr)
        if self.backend == "haar":
            return self._detect_haar(frame_bgr)
        return self._detect_dnn(frame_bgr)

    def _init_haar(self) -> None:
        filename = "haarcascade_frontalface_default.xml"
        candidates = []

        cv2_data = getattr(cv2, "data", None)
        if cv2_data is not None:
            haar_root = getattr(cv2_data, "haarcascades", None)
            if haar_root:
                candidates.append(Path(haar_root) / filename)

        cv2_file = getattr(cv2, "__file__", None)
        if cv2_file:
            cv2_root = Path(cv2_file).resolve().parent
            candidates.extend(
                [
                    cv2_root / "data" / "haarcascades" / filename,
                    cv2_root.parent / "share" / "opencv4" / "haarcascades" / filename,
                    cv2_root.parent / "share" / "opencv" / "haarcascades" / filename,
                ]
            )

        cascade_path = None
        for candidate in candidates:
            if candidate.exists():
                cascade_path = candidate
                break
        if cascade_path is None:
            raise FileNotFoundError(
                "Unable to locate Haar cascade XML. Install opencv-data or configure a detector backend with models."
            )

        self.haar = cv2.CascadeClassifier(str(cascade_path))
        if self.haar.empty():
            raise RuntimeError(f"Failed to initialize Haar cascade at {cascade_path}")

    def _detect_mediapipe(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        assert self.face_detection is not None
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.face_detection.process(rgb)
        out: List[FaceDetection] = []
        if not result.detections:
            return out

        for det in result.detections:
            score = float(det.score[0]) if det.score else 0.0
            if score < self.min_conf:
                continue
            rel = det.location_data.relative_bounding_box
            x = max(0, int(rel.xmin * w))
            y = max(0, int(rel.ymin * h))
            bw = max(1, int(rel.width * w))
            bh = max(1, int(rel.height * h))
            if bw < self.min_face_size or bh < self.min_face_size:
                continue
            out.append(FaceDetection((x, y, bw, bh), score))
        return out

    def _detect_dnn(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        assert self.dnn_net is not None
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        out: List[FaceDetection] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.min_conf:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            if bw < self.min_face_size or bh < self.min_face_size:
                continue
            out.append(FaceDetection((x1, y1, bw, bh), conf))
        return out

    def _detect_haar(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        assert self.haar is not None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        out: List[FaceDetection] = []
        for x, y, w, h in faces:
            out.append(FaceDetection((int(x), int(y), int(w), int(h)), 1.0))
        return out

