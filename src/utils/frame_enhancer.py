from __future__ import annotations

import cv2
import numpy as np


class PodcastFrameEnhancer:
    """
    Podcast-focused frame enhancement stack using only OpenCV:
      - gray-world white balance
      - CLAHE local contrast enhancement
      - light denoising
      - unsharp-mask detail boost
    Each stage is configurable and can be toggled from config.
    """

    def __init__(self, config: dict, logger) -> None:
        self.logger = logger
        self.enabled = bool(config.get("enabled", True))
        self.white_balance = bool(config.get("white_balance", True))
        self.clahe_enabled = bool(config.get("clahe", True))
        self.denoise_enabled = bool(config.get("denoise", True))
        self.sharpen_enabled = bool(config.get("sharpen", True))

        self.clahe_clip = float(config.get("clahe_clip_limit", 2.0))
        self.clahe_grid = int(config.get("clahe_tile_grid_size", 8))
        self.denoise_strength = int(config.get("denoise_strength", 5))
        self.sharpen_amount = float(config.get("sharpen_amount", 1.2))
        self.sharpen_sigma = float(config.get("sharpen_sigma", 1.2))

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame_bgr

        out = frame_bgr
        if self.white_balance:
            out = self._gray_world_white_balance(out)
        if self.clahe_enabled:
            out = self._apply_clahe(out)
        if self.denoise_enabled:
            out = self._denoise(out)
        if self.sharpen_enabled:
            out = self._unsharp_mask(out)
        return out

    @staticmethod
    def _gray_world_white_balance(frame_bgr: np.ndarray) -> np.ndarray:
        bgr = frame_bgr.astype(np.float32)
        mean_b = float(np.mean(bgr[:, :, 0])) + 1e-6
        mean_g = float(np.mean(bgr[:, :, 1])) + 1e-6
        mean_r = float(np.mean(bgr[:, :, 2])) + 1e-6
        mean_gray = (mean_b + mean_g + mean_r) / 3.0

        bgr[:, :, 0] *= mean_gray / mean_b
        bgr[:, :, 1] *= mean_gray / mean_g
        bgr[:, :, 2] *= mean_gray / mean_r
        return np.clip(bgr, 0, 255).astype(np.uint8)

    def _apply_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))
        l2 = clahe.apply(l)
        merged = cv2.merge((l2, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _denoise(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Bilateral filter preserves edges (important for face/body boundaries).
        d = max(3, int(self.denoise_strength))
        sigma = max(10, int(self.denoise_strength * 8))
        return cv2.bilateralFilter(frame_bgr, d=d, sigmaColor=sigma, sigmaSpace=sigma)

    def _unsharp_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame_bgr, (0, 0), self.sharpen_sigma)
        sharp = cv2.addWeighted(
            frame_bgr,
            1.0 + self.sharpen_amount,
            blurred,
            -self.sharpen_amount,
            0,
        )
        return np.clip(sharp, 0, 255).astype(np.uint8)

