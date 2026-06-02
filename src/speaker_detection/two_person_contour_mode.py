from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .face_detection import FaceDetection, FaceDetector

BBox = Tuple[int, int, int, int]


@dataclass
class TwoPersonResult:
    contours: List[np.ndarray]
    ordered_faces: List[FaceDetection]


class TwoPersonContourProcessor:
    def __init__(self, config: Dict, logger: logging.Logger, detector: FaceDetector | None = None) -> None:
        self.logger = logger
        self.mode_cfg = config.get("two_person_contour_mode", {})
        self.speaker_cfg = {**config.get("speaker_detection", {})}

        # Allow two-person contour mode to override detector settings without touching main speaker config.
        for key in ("detector_backend", "detection_confidence", "dnn_prototxt", "dnn_model", "min_face_size"):
            if key in self.mode_cfg:
                self.speaker_cfg[key] = self.mode_cfg[key]

        self.contour_color = self._color_bgr(self.mode_cfg.get("contour_color", {"r": 255, "g": 255, "b": 255}))
        self.contour_thickness = int(self.mode_cfg.get("contour_thickness", 6))
        self.expand_x = float(self.mode_cfg.get("expand_x", 1.4))
        self.expand_top = float(self.mode_cfg.get("expand_top", 0.35))
        self.expand_bottom = float(self.mode_cfg.get("expand_bottom", 3.0))
        self.min_area = int(self.mode_cfg.get("min_area", 1200))
        self.fail_on_less_than_two = bool(self.mode_cfg.get("fail_on_less_than_two", True))
        self.duplicate_iou_threshold = float(self.mode_cfg.get("duplicate_iou_threshold", 0.35))
        self.min_face_center_distance_ratio = float(self.mode_cfg.get("min_face_center_distance_ratio", 0.6))
        self.shadow_gate_pad_x_ratio = float(self.mode_cfg.get("shadow_gate_pad_x_ratio", 0.7))
        self.shadow_gate_pad_top_ratio = float(self.mode_cfg.get("shadow_gate_pad_top_ratio", 0.2))
        self.shadow_gate_pad_bottom_ratio = float(self.mode_cfg.get("shadow_gate_pad_bottom_ratio", 0.35))

        self.detector = detector or FaceDetector(self.speaker_cfg, logger)

    @staticmethod
    def _color_bgr(color_cfg: dict) -> Tuple[int, int, int]:
        return (
            int(color_cfg.get("b", 255)),
            int(color_cfg.get("g", 255)),
            int(color_cfg.get("r", 255)),
        )

    @staticmethod
    def _center_x(bbox: BBox) -> float:
        x, _, w, _ = bbox
        return x + (w / 2.0)

    @staticmethod
    def _score(det: FaceDetection) -> float:
        _, _, w, h = det.bbox
        return float(det.confidence) * float(w * h)

    @staticmethod
    def _center(bbox: BBox) -> Tuple[float, float]:
        x, y, w, h = bbox
        return x + (w / 2.0), y + (h / 2.0)

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        union = float((aw * ah) + (bw * bh)) - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _is_duplicate_face(self, candidate: FaceDetection, picked: List[FaceDetection]) -> bool:
        cx, cy = self._center(candidate.bbox)
        _, _, cw, ch = candidate.bbox
        for chosen in picked:
            if self._iou(candidate.bbox, chosen.bbox) >= self.duplicate_iou_threshold:
                return True
            sx, sy = self._center(chosen.bbox)
            _, _, sw, sh = chosen.bbox
            # Use the smallest face dimension across both boxes so near-identical overlapping detections
            # of the same person are filtered even when one box is slightly larger.
            min_size = max(1.0, float(min(cw, ch, sw, sh)))
            min_dist = self.min_face_center_distance_ratio * min_size
            dist_sq = ((cx - sx) ** 2) + ((cy - sy) ** 2)
            min_dist_sq = min_dist**2
            if dist_sq < min_dist_sq:
                return True
        return False

    def _pick_two_faces(self, detections: Iterable[FaceDetection]) -> List[FaceDetection]:
        ranked = sorted(detections, key=self._score, reverse=True)
        picked: List[FaceDetection] = []
        for det in ranked:
            if self._is_duplicate_face(det, picked):
                continue
            picked.append(det)
            if len(picked) == 2:
                break
        return sorted(picked, key=lambda d: self._center_x(d.bbox))

    def _expanded_roi(self, frame_shape: Tuple[int, int, int], face_bbox: BBox) -> Tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        x, y, bw, bh = face_bbox
        cx = x + bw / 2.0

        ex1 = int(cx - (bw * self.expand_x))
        ex2 = int(cx + (bw * self.expand_x))
        ey1 = int(y - bh * self.expand_top)
        ey2 = int(y + bh * self.expand_bottom)

        ex1 = max(0, ex1)
        ey1 = max(0, ey1)
        ex2 = min(w, max(ex1 + 2, ex2))
        ey2 = min(h, max(ey1 + 2, ey2))
        return ex1, ey1, ex2, ey2

    def _fallback_contour(self, frame_shape: Tuple[int, int, int], bbox: BBox) -> np.ndarray | None:
        h, w = frame_shape[:2]
        x, y, bw, bh = bbox
        cx = x + bw // 2

        body_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            body_mask,
            (cx, y + bh // 2),
            (max(8, int(0.65 * bw)), max(8, int(0.75 * bh))),
            0,
            0,
            360,
            255,
            -1,
        )

        shoulder_y = min(h - 1, y + int(1.2 * bh))
        bottom_y = min(h - 1, y + int(3.0 * bh))
        torso_half_w = max(12, int(1.25 * bw))
        pts = np.array(
            [
                [max(0, cx - int(0.8 * bw)), shoulder_y],
                [max(0, cx - torso_half_w), min(h - 1, shoulder_y + int(0.35 * bh))],
                [max(0, cx - int(0.85 * torso_half_w)), bottom_y],
                [min(w - 1, cx + int(0.85 * torso_half_w)), bottom_y],
                [min(w - 1, cx + torso_half_w), min(h - 1, shoulder_y + int(0.35 * bh))],
                [min(w - 1, cx + int(0.8 * bw)), shoulder_y],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(body_mask, [pts], 255)
        contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _mask_from_roi(self, roi_bgr: np.ndarray, local_face_bbox: BBox) -> np.ndarray:
        roi_h, roi_w = roi_bgr.shape[:2]
        x, y, w, h = local_face_bbox

        mask = np.full((roi_h, roi_w), cv2.GC_BGD, dtype=np.uint8)
        mask[y : y + h, x : x + w] = cv2.GC_PR_FGD

        torso_y2 = min(roi_h, y + int(3.0 * h))
        torso_x1 = max(0, x - int(0.8 * w))
        torso_x2 = min(roi_w, x + int(1.8 * w))
        mask[y:torso_y2, torso_x1:torso_x2] = cv2.GC_PR_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(roi_bgr, mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
            fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            fg = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.rectangle(fg, (torso_x1, y), (max(torso_x1 + 2, torso_x2), max(y + 2, torso_y2)), 255, -1)
            cv2.ellipse(
                fg,
                (x + w // 2, y + h // 2),
                (max(2, int(0.7 * w)), max(2, int(0.8 * h))),
                0,
                0,
                360,
                255,
                -1,
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        return self._gate_shadow_bleed(fg, local_face_bbox)

    def _gate_shadow_bleed(self, fg_mask: np.ndarray, local_face_bbox: BBox) -> np.ndarray:
        roi_h, roi_w = fg_mask.shape[:2]
        x, y, w, h = local_face_bbox
        cx = x + (w // 2)

        prior = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.ellipse(
            prior,
            (cx, y + h // 2),
            (max(4, int(0.7 * w)), max(4, int(0.85 * h))),
            0,
            0,
            360,
            255,
            -1,
        )

        shoulder_y = min(roi_h - 1, y + int(1.15 * h))
        bottom_y = min(roi_h - 1, y + int(3.1 * h))
        torso_half_w = max(8, int(1.2 * w))
        pts = np.array(
            [
                [max(0, cx - int(0.8 * w)), shoulder_y],
                [max(0, cx - torso_half_w), min(roi_h - 1, shoulder_y + int(0.35 * h))],
                [max(0, cx - int(0.9 * torso_half_w)), bottom_y],
                [min(roi_w - 1, cx + int(0.9 * torso_half_w)), bottom_y],
                [min(roi_w - 1, cx + torso_half_w), min(roi_h - 1, shoulder_y + int(0.35 * h))],
                [min(roi_w - 1, cx + int(0.8 * w)), shoulder_y],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(prior, [pts], 255)

        pad_x = max(2, int(self.shadow_gate_pad_x_ratio * w))
        pad_top = max(1, int(self.shadow_gate_pad_top_ratio * h))
        pad_bottom = max(1, int(self.shadow_gate_pad_bottom_ratio * h))
        gate_x1 = max(0, x - pad_x)
        gate_x2 = min(roi_w, x + w + pad_x)
        gate_y1 = max(0, y - pad_top)
        gate_y2 = min(roi_h, y + int(3.2 * h) + pad_bottom)
        cv2.rectangle(prior, (gate_x1, gate_y1), (max(gate_x1 + 2, gate_x2), max(gate_y1 + 2, gate_y2)), 255, -1)

        gated = cv2.bitwise_and(fg_mask, prior)
        # Always preserve the face seed region.
        gated[y : y + h, x : x + w] = cv2.bitwise_or(gated[y : y + h, x : x + w], fg_mask[y : y + h, x : x + w])

        if cv2.countNonZero(gated) >= max(40, int(0.6 * w * h)):
            return gated
        return fg_mask

    def _contour_for_face(self, frame_bgr: np.ndarray, face_bbox: BBox) -> np.ndarray | None:
        ex1, ey1, ex2, ey2 = self._expanded_roi(frame_bgr.shape, face_bbox)
        roi = frame_bgr[ey1:ey2, ex1:ex2]
        if roi.size == 0:
            return self._fallback_contour(frame_bgr.shape, face_bbox)

        lx = max(0, face_bbox[0] - ex1)
        ly = max(0, face_bbox[1] - ey1)
        lw = max(2, min(roi.shape[1] - lx, face_bbox[2]))
        lh = max(2, min(roi.shape[0] - ly, face_bbox[3]))
        local_face = (lx, ly, lw, lh)
        fg_mask = self._mask_from_roi(roi, local_face)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            valid = [c for c in contours if cv2.contourArea(c) >= self.min_area]
            if valid:
                face_center = (float(lx + (lw / 2.0)), float(ly + (lh / 2.0)))
                anchored = [c for c in valid if cv2.pointPolygonTest(c, face_center, False) >= 0]
                contour = max(anchored if anchored else valid, key=cv2.contourArea)
                return contour + np.array([[[ex1, ey1]]], dtype=np.int32)

        return self._fallback_contour(frame_bgr.shape, face_bbox)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, TwoPersonResult]:
        detections = self.detector.detect(frame_bgr)
        faces = self._pick_two_faces(detections)

        if len(faces) < 2:
            message = f"Expected 2 people, found {len(faces)} face(s)."
            if self.fail_on_less_than_two:
                raise RuntimeError(message)
            self.logger.warning(message)

        contours: List[np.ndarray] = []
        for face in faces:
            contour = self._contour_for_face(frame_bgr, face.bbox)
            if contour is not None:
                contours.append(contour)

        rendered = frame_bgr.copy()
        for contour in contours:
            cv2.drawContours(rendered, [contour], -1, self.contour_color, self.contour_thickness, lineType=cv2.LINE_AA)

        return rendered, TwoPersonResult(contours=contours, ordered_faces=faces)

    def process_image_file(self, input_path: Path, output_path: Path) -> TwoPersonResult:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {input_path}")
        rendered, result = self.process_frame(image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), rendered):
            raise RuntimeError(f"Failed to write output image: {output_path}")
        return result


def run_single_image_mode(config: Dict, input_image: Path, output_image: Path, logger: logging.Logger) -> TwoPersonResult:
    processor = TwoPersonContourProcessor(config=config, logger=logger)
    return processor.process_image_file(input_image, output_image)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-image two-person contour mode.")
    parser.add_argument("--input-image", required=True, help="Input image path")
    parser.add_argument("--output-image", required=True, help="Output image path")
    parser.add_argument("--config", default="config/config.yaml", help="Config YAML path")
    return parser


def main() -> int:
    from src.utils import load_config, setup_logger

    args = _build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / args.config)
    logger = setup_logger(project_root / "logs")

    result = run_single_image_mode(
        config=config,
        input_image=Path(args.input_image).expanduser().resolve(),
        output_image=Path(args.output_image).expanduser().resolve(),
        logger=logger,
    )
    logger.info("Two-person image contour complete. Contours drawn: %d", len(result.contours))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
