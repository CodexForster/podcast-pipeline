from pathlib import Path

from src.utils.config_loader import load_config


def test_all_toggles_present_and_default_true():
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    cfg = load_config(config_path)

    for section in ("ingest", "preprocessing", "speaker_detection"):
        assert cfg["pipeline"][section]["enabled"] is True

