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

