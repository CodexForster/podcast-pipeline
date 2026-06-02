# Podcast Pipeline (Steps 1-3)

Production-oriented Python pipeline for preprocessing multi-camera podcast footage using OpenCV + FFmpeg.

Implemented scope:
- Step 1: Ingest pipeline
- Step 2: Preprocessing pipeline
- Step 3: Speaker detection pipeline

All steps/substeps are controlled by one central config file: `config/config.yaml`.

## Project Structure

```text
podcast_pipeline/
├── config/
│   └── config.yaml
├── src/
│   ├── ingest/
│   ├── preprocessing/
│   ├── speaker_detection/
│   ├── utils/
│   └── main.py
├── tests/
├── outputs/
├── logs/
├── main.py
└── requirements.txt
```

## Requirements

- Python 3.11+
- FFmpeg and ffprobe available in PATH

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Run

From project root:

```bash
python main.py --input ./raw_videos/
```

Optional inputs:

```bash
python main.py \
  --input ./raw_videos/ \
  --audio-dir ./audio/ \
  --overlay-dir ./overlays/ \
  --intro ./assets/intro.mp4 \
  --outro ./assets/outro.mp4
```

## Config-Driven Toggles

All toggles default to `true` in `config/config.yaml`.

Examples:
- Disable full preprocessing step:
  - `pipeline.preprocessing.enabled: false`
- Disable only frame extraction:
  - `pipeline.preprocessing.extract_frames: false`
- Disable speaker borders:
  - `pipeline.speaker_detection.draw_speaker_border: false`

## Step Details

### 1) Ingest
- Validates input assets (single file or folder scan)
- Uses FFmpeg validation for corruption checks
- Extracts video metadata with `cv2.VideoCapture`
- Organizes assets into:
  - `raw/`, `processed/`, `proxies/`, `frames/`, `logs/`, `outputs/`

### 2) Preprocessing
- FPS normalization (FFmpeg)
- Resize to target resolution (FFmpeg)
- Proxy generation (FFmpeg)
- Frame extraction to `frames/` (OpenCV)
- Audio normalization using configurable FFmpeg filter (`loudnorm` default)

### 3) Speaker Detection
- Face detection: MediaPipe (default) or OpenCV DNN (configurable)
- Face tracking: centroid tracker with consistent speaker IDs
- Speaker activity estimation:
  - lip movement (Face Mesh if available, ROI fallback)
  - upper-body/head motion from frame differencing
  - temporal smoothing
- Podcast visual enhancement stack (white balance, CLAHE, denoise, sharpen)
- Active speaker silhouette rendering (thick white contour around face/body boundary, no square box labels)

## Notes / Limitations

- OpenCV DNN backend requires external Caffe model files (`dnn_prototxt`, `dnn_model`).
- MediaPipe is used by default for easier setup without manual face model download.
- Lip activity and speaker identification are heuristic and can be affected by occlusion, profile faces, and low light.
- This code is designed to be extended for later steps (auto-editing, shot selection, timeline generation).

## Single-Image Two-Person Contour Mode

For debugging on one image (two people conversing), run:

```bash
python -m src.speaker_detection.two_person_contour_mode \
  --input-image /absolute/path/to/input.jpg \
  --output-image /absolute/path/to/output.jpg \
  --config config/config.yaml
```

Tune behavior in `two_person_contour_mode` inside `config/config.yaml`.
Set `two_person_contour_mode.enabled: true` to reuse the same contour logic frame-by-frame in the video speaker-detection pipeline.
