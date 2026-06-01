from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    _ensure_toggle_defaults(config)
    return config


def _ensure_toggle_defaults(config: Dict[str, Any]) -> None:
    pipeline = config.setdefault("pipeline", {})
    two_person_mode = config.setdefault("two_person_contour_mode", {})
    ingest = pipeline.setdefault("ingest", {})
    preprocessing = pipeline.setdefault("preprocessing", {})
    speaker = pipeline.setdefault("speaker_detection", {})

    ingest_defaults = {
        "enabled": True,
        "validate_inputs": True,
        "extract_metadata": True,
        "organize_files": True,
    }
    preprocessing_defaults = {
        "enabled": True,
        "normalize_framerate": True,
        "resize_video": True,
        "generate_proxies": True,
        "extract_frames": True,
        "audio_normalization": True,
    }
    speaker_defaults = {
        "enabled": True,
        "face_detection": True,
        "face_tracking": True,
        "speaker_identification": True,
        "lip_movement_detection": True,
        "motion_detection": True,
        "podcast_visual_enhancement": True,
        "draw_speaker_border": True,
    }

    for key, value in ingest_defaults.items():
        ingest.setdefault(key, value)
    for key, value in preprocessing_defaults.items():
        preprocessing.setdefault(key, value)
    for key, value in speaker_defaults.items():
        speaker.setdefault(key, value)

    image_defaults = {
        "enabled": False,
        "detector_backend": speaker.get("detector_backend", "mediapipe"),
        "detection_confidence": speaker.get("detection_confidence", 0.5),
        "min_face_size": speaker.get("min_face_size", 32),
        "fail_on_less_than_two": True,
        "contour_thickness": 6,
        "contour_color": {"r": 255, "g": 255, "b": 255},
        "expand_x": 1.4,
        "expand_top": 0.35,
        "expand_bottom": 3.0,
        "min_area": 1200,
    }
    for key, value in image_defaults.items():
        two_person_mode.setdefault(key, value)

