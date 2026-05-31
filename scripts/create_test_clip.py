from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a short test clip from a longer video.")
    parser.add_argument(
        "--input",
        default="../input/20260403_recording.mp4",
        help="Input video path (default: ../input/20260403_recording.mp4 relative to this script).",
    )
    parser.add_argument(
        "--output",
        default="../input/test3min.mp4",
        help="Output clip path (default: ../input/test3min.mp4 relative to this script).",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=180,
        help="Duration of output clip in seconds (default: 180).",
    )
    return parser


def create_clip(input_path: Path, output_path: Path, duration_seconds: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if duration_seconds <= 0:
        raise ValueError("duration-seconds must be > 0")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-t",
        str(duration_seconds),
        "-c",
        "copy",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed.\nSTDERR:\n{proc.stderr}")


def main() -> int:
    args = build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = (script_dir / args.input).resolve()
    output_path = (script_dir / args.output).resolve()

    create_clip(input_path=input_path, output_path=output_path, duration_seconds=args.duration_seconds)
    print(f"[DONE] Created clip: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

