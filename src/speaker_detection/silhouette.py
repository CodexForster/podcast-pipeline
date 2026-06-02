from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

from .tracking import BBox


@dataclass
class _MaskCache:
    mask: np.ndarray
    roi_shape: Tuple[int, int]


class SilhouetteRenderer:
    """
    Draws editorial silhouette borders (face + upper body) for visible speakers.
    Uses GrabCut-based mask extraction with a robust geometric fallback.
    """

    def __init__(self, config: dict, logger) -> None:
        self.logger = logger
        self.enabled = bool(config.get("enabled", True))
        self.outline_all_speakers = bool(config.get("outline_all_speakers", True))
        self.thickness = int(config.get("thickness", 6))
        self.active_thickness_boost = int(config.get("active_thickness_boost", 2))
        self.shadow_thickness = int(config.get("shadow_thickness", 2))
        self.shadow_color = self._color_bgr(config.get("shadow_color", {"r": 0, "g": 0, "b": 0}))
        self.expand_x = float(config.get("expand_x", 1.4))
        self.expand_top = float(config.get("expand_top", 0.35))
        self.expand_bottom = float(config.get("expand_bottom", 3.0))
        self.min_area = int(config.get("min_area", 1200))
        self.max_roi_height = int(config.get("max_roi_height", 320))
        self.smoothing_eps = float(config.get("smoothing_eps", 0.008))
        self.color = self._color_bgr(config.get("color", {"r": 255, "g": 255, "b": 255}))
        self.shadow_gate_pad_x_ratio = float(config.get("shadow_gate_pad_x_ratio", 0.7))
        self.shadow_gate_pad_top_ratio = float(config.get("shadow_gate_pad_top_ratio", 0.2))
        self.shadow_gate_pad_bottom_ratio = float(config.get("shadow_gate_pad_bottom_ratio", 0.35))
        self.mask_cache: Dict[int, _MaskCache] = {}

    @staticmethod
    def _color_bgr(color_cfg: dict) -> Tuple[int, int, int]:
        r = int(color_cfg.get("r", 255))
        g = int(color_cfg.get("g", 255))
        b = int(color_cfg.get("b", 255))
        return (b, g, r)

    def draw_speakers(self, frame_bgr: np.ndarray, tracks: Dict[int, object], active_id: int | None) -> np.ndarray:
        if not self.enabled or not tracks:
            return frame_bgr

        out = frame_bgr.copy()
        items = tracks.items() if self.outline_all_speakers else [(active_id, tracks[active_id])] if active_id in tracks else []
        for sid, track in items:
            contour = self._speaker_contour(out, sid, track.bbox)
            if contour is None:
                continue
            contour = self._smooth_contour(contour)
            thickness = self.thickness + (self.active_thickness_boost if sid == active_id else 0)
            self._draw_editorial_contour(out, contour, thickness)
        return out

    def _speaker_contour(self, frame_bgr: np.ndarray, speaker_id: int, bbox: BBox):
        roi_rect = self._expanded_roi(frame_bgr.shape, bbox)
        x1, y1, x2, y2 = roi_rect
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        local_bbox = (bbox[0] - x1, bbox[1] - y1, bbox[2], bbox[3])
        fg_mask = self._estimate_foreground_mask(speaker_id, roi, active_bbox=local_bbox)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) >= self.min_area:
                return contour + np.array([[[x1, y1]]], dtype=np.int32)

        # Fallback guarantees visible silhouette if segmentation is weak.
        return self._fallback_body_contour(frame_bgr.shape, bbox)

    def _expanded_roi(self, frame_shape, face_bbox: BBox) -> Tuple[int, int, int, int]:
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

    def _fallback_body_contour(self, frame_shape, face_bbox: BBox):
        h, w = frame_shape[:2]
        x, y, bw, bh = face_bbox
        cx = x + bw // 2

        head_rx = max(8, int(0.65 * bw))
        head_ry = max(8, int(0.75 * bh))
        top_y = max(0, y - int(0.15 * bh))
        bottom_y = min(h - 1, y + int(3.0 * bh))
        shoulder_y = min(h - 1, y + int(1.2 * bh))
        torso_half_w = max(12, int(1.25 * bw))

        canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(canvas, (cx, y + bh // 2), (head_rx, head_ry), 0, 0, 360, 255, -1)
        pts = np.array(
            [
                [max(0, cx - int(0.8 * head_rx)), shoulder_y],
                [max(0, cx - torso_half_w), min(h - 1, shoulder_y + int(0.35 * bh))],
                [max(0, cx - int(0.85 * torso_half_w)), bottom_y],
                [min(w - 1, cx + int(0.85 * torso_half_w)), bottom_y],
                [min(w - 1, cx + torso_half_w), min(h - 1, shoulder_y + int(0.35 * bh))],
                [min(w - 1, cx + int(0.8 * head_rx)), shoulder_y],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [pts], 255)
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _smooth_contour(self, contour: np.ndarray) -> np.ndarray:
        peri = cv2.arcLength(contour, True)
        eps = max(1.0, self.smoothing_eps * peri)
        approx = cv2.approxPolyDP(contour, eps, True)
        return approx if approx is not None and len(approx) >= 3 else contour

    def _draw_editorial_contour(self, frame: np.ndarray, contour: np.ndarray, thickness: int) -> None:
        shadow_thickness = max(1, thickness + self.shadow_thickness)
        cv2.drawContours(frame, [contour], -1, self.shadow_color, shadow_thickness, lineType=cv2.LINE_AA)
        cv2.drawContours(frame, [contour], -1, self.color, max(1, thickness), lineType=cv2.LINE_AA)

    def _estimate_foreground_mask(self, speaker_id: int, roi_bgr: np.ndarray, active_bbox: BBox) -> np.ndarray:
        roi_h, roi_w = roi_bgr.shape[:2]

        # Downscale ROI to keep grabCut fast.
        scale = 1.0
        if roi_h > self.max_roi_height:
            scale = self.max_roi_height / float(roi_h)
        small = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale < 1.0 else roi_bgr
        sh, sw = small.shape[:2]

        sx, sy, sbw, sbh = [int(round(v * scale)) for v in active_bbox]
        sx = max(0, min(sw - 1, sx))
        sy = max(0, min(sh - 1, sy))
        sbw = max(2, min(sw - sx, sbw))
        sbh = max(2, min(sh - sy, sbh))

        mask = np.full((sh, sw), cv2.GC_BGD, dtype=np.uint8)
        # Face region and small torso region are probable foreground seeds.
        mask[sy : sy + sbh, sx : sx + sbw] = cv2.GC_PR_FGD
        torso_y2 = min(sh, sy + int(3.2 * sbh))
        torso_x1 = max(0, sx - int(0.9 * sbw))
        torso_x2 = min(sw, sx + int(1.9 * sbw))
        mask[sy:torso_y2, torso_x1:torso_x2] = cv2.GC_PR_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cache = self.mask_cache.get(speaker_id)
        if cache is not None and cache.roi_shape == (sh, sw):
            prior = np.where(cache.mask > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
            prior[sy : sy + sbh, sx : sx + sbw] = cv2.GC_FGD
            mask = prior
            mode = cv2.GC_INIT_WITH_MASK
            iter_count = 1
        else:
            rect = (
                max(0, torso_x1),
                max(0, sy),
                max(2, torso_x2 - torso_x1),
                max(2, torso_y2 - sy),
            )
            mode = cv2.GC_INIT_WITH_RECT
            iter_count = 2

        try:
            cv2.grabCut(
                small,
                mask,
                rect if mode == cv2.GC_INIT_WITH_RECT else None,
                bgd_model,
                fgd_model,
                iter_count,
                mode,
            )
        except cv2.error:
            # GrabCut can fail on low-contrast or degenerate regions.
            # Return a deterministic torso fallback mask so outlines still render.
            fallback = np.zeros((sh, sw), dtype=np.uint8)
            cv2.rectangle(fallback, (torso_x1, sy), (max(torso_x1 + 2, torso_x2), max(sy + 2, torso_y2)), 255, -1)
            cv2.ellipse(
                fallback,
                (sx + sbw // 2, sy + sbh // 2),
                (max(2, int(0.7 * sbw)), max(2, int(0.8 * sbh))),
                0,
                0,
                360,
                255,
                -1,
            )
            fg_small = fallback
            self.mask_cache[speaker_id] = _MaskCache(mask=(fg_small > 0).astype(np.uint8), roi_shape=(sh, sw))
            if scale < 1.0:
                fg = cv2.resize(fg_small, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
                _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
                return fg
            return fg_small
        fg_small = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # Clean mask and keep largest connected component.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_small = cv2.morphologyEx(fg_small, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_small = cv2.morphologyEx(fg_small, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_small = self._gate_shadow_bleed(fg_small, (sx, sy, sbw, sbh))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_small)
        if num_labels > 1:
            areas = stats[:, cv2.CC_STAT_AREA]
            largest = int(np.argmax(areas[1:]) + 1)
            fg_small = np.where(labels == largest, 255, 0).astype(np.uint8)

        self.mask_cache[speaker_id] = _MaskCache(mask=(fg_small > 0).astype(np.uint8), roi_shape=(sh, sw))
        if scale < 1.0:
            fg = cv2.resize(fg_small, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
            return fg
        return fg_small

    def _gate_shadow_bleed(self, fg_mask: np.ndarray, active_bbox: BBox) -> np.ndarray:
        h, w = fg_mask.shape[:2]
        x, y, bw, bh = active_bbox
        cx = x + (bw // 2)

        prior = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            prior,
            (cx, y + bh // 2),
            (max(4, int(0.7 * bw)), max(4, int(0.85 * bh))),
            0,
            0,
            360,
            255,
            -1,
        )

        shoulder_y = min(h - 1, y + int(1.15 * bh))
        bottom_y = min(h - 1, y + int(3.1 * bh))
        torso_half_w = max(8, int(1.2 * bw))
        pts = np.array(
            [
                [max(0, cx - int(0.8 * bw)), shoulder_y],
                [max(0, cx - torso_half_w), min(h - 1, shoulder_y + int(0.35 * bh))],
                [max(0, cx - int(0.9 * torso_half_w)), bottom_y],
                [min(w - 1, cx + int(0.9 * torso_half_w)), bottom_y],
                [min(w - 1, cx + torso_half_w), min(h - 1, shoulder_y + int(0.35 * bh))],
                [min(w - 1, cx + int(0.8 * bw)), shoulder_y],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(prior, [pts], 255)

        pad_x = max(2, int(self.shadow_gate_pad_x_ratio * bw))
        pad_top = max(1, int(self.shadow_gate_pad_top_ratio * bh))
        pad_bottom = max(1, int(self.shadow_gate_pad_bottom_ratio * bh))
        gate_x1 = max(0, x - pad_x)
        gate_x2 = min(w, x + bw + pad_x)
        gate_y1 = max(0, y - pad_top)
        gate_y2 = min(h, y + int(3.2 * bh) + pad_bottom)
        cv2.rectangle(prior, (gate_x1, gate_y1), (max(gate_x1 + 2, gate_x2), max(gate_y1 + 2, gate_y2)), 255, -1)

        gated = cv2.bitwise_and(fg_mask, prior)
        gated[y : y + bh, x : x + bw] = cv2.bitwise_or(gated[y : y + bh, x : x + bw], fg_mask[y : y + bh, x : x + bw])

        if cv2.countNonZero(gated) >= max(40, int(0.6 * bw * bh)):
            return gated
        return fg_mask
