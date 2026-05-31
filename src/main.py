from __future__ import annotations

import argparse
from pathlib import Path

from ingest import IngestPipeline
from preprocessing import PreprocessingPipeline
from speaker_detection import SpeakerDetectionPipeline
from utils import ensure_structure, load_config, setup_logger
from utils.ffmpeg_utils import ensure_ffmpeg_available


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video podcast preprocessing pipeline (Steps 1-3).")
    parser.add_argument("--input", required=True, help="Input file or directory containing raw videos.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--audio-dir", default=None, help="Optional directory with external audio files.")
    parser.add_argument("--overlay-dir", default=None, help="Optional directory with overlay assets.")
    parser.add_argument("--intro", default=None, help="Optional intro media file.")
    parser.add_argument("--outro", default=None, help="Optional outro media file.")
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / args.config)
    directories = ensure_structure(project_root, config.get("paths", {}))
    logger = setup_logger(directories["logs_dir"])
    ensure_ffmpeg_available()

    ingest = IngestPipeline(config, directories, logger)
    preprocess = PreprocessingPipeline(config, directories, logger)
    speaker = SpeakerDetectionPipeline(config, directories, logger)

    ingest_result = ingest.run(
        input_path=Path(args.input),
        audio_dir=Path(args.audio_dir) if args.audio_dir else None,
        overlay_dir=Path(args.overlay_dir) if args.overlay_dir else None,
        intro_path=Path(args.intro) if args.intro else None,
        outro_path=Path(args.outro) if args.outro else None,
    )

    preprocess_result = preprocess.run(ingest_result.videos)
    final_outputs = speaker.run(preprocess_result.processed_videos)
    logger.info("Pipeline complete. Final outputs: %s", [str(p) for p in final_outputs])


def main() -> int:
    args = build_parser().parse_args()
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

