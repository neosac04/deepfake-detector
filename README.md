# Deepfake Detector (Starter Scaffold)

This project is a clean starter scaffold built by combining practical ideas from:
- `Deepfake_detection_using_deep_learning-master` (strong PyTorch ResNeXt+LSTM approach + face-crop sequence pipeline)
- `DeepFake_Detection-main` (deployment/UI intent and CNN+RNN model experimentation mindset)

## Review Summary of Source Repos
- **Repo 1 strengths:** usable model logic (ResNeXt + LSTM), frame/face preprocessing, end-to-end inference flow.
- **Repo 1 gaps:** tightly coupled notebooks + Django code, hard-coded paths, mixed concerns, heavy legacy dependencies.
- **Repo 2 strengths:** clear product intent (upload/analyze), alternative model family ideas (EfficientNet/Inception + GRU), deployment orientation.
- **Repo 2 gaps:** incomplete backend app logic, mostly notebook artifacts, non-production package structure.

## What this scaffold gives you
- Modular `src/` package
- Config-driven training and inference
- Reusable data preprocessing and dataset classes
- Two starter model modules:
  - `ResNeXtLSTM` (primary, from repo-1 direction)
  - `EfficientNetGRU` (optional baseline inspired by repo-2 direction)
- FastAPI starter endpoint for upload + prediction
- Simple CLI scripts for data prep, training, and inference

## Project Structure
```text
deepfake-detector/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
│   ├── checkpoints/
│   └── exported/
├── notebooks/
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── infer.py
├── src/deepfake_detector/
│   ├── api/
│   │   └── app.py
│   ├── data/
│   │   ├── video.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── resnext_lstm.py
│   │   └── efficientnet_gru.py
│   ├── pipelines/
│   │   ├── train_pipeline.py
│   │   └── inference_pipeline.py
│   ├── utils/
│   │   └── logging.py
│   ├── config.py
│   └── __init__.py
├── tests/
│   └── test_config.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Quick Start
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare a metadata CSV with columns:
   - `video_path`
   - `label` (`0` for FAKE, `1` for REAL)
4. Run training:
   ```bash
   python scripts/train.py --config configs/default.yaml --metadata_csv path/to/metadata.csv --output models/checkpoints/baseline.pt
   ```
5. Run inference:
   ```bash
   python scripts/infer.py --config configs/default.yaml --checkpoint models/checkpoints/baseline.pt --video path/to/video.mp4
   ```
   Or pass a still image instead:
   ```bash
   python scripts/infer.py --config configs/default.yaml --checkpoint models/checkpoints/baseline.pt --image path/to/image.jpg
   ```
6. Run API (starter):
   ```bash
   PYTHONPATH=src uvicorn deepfake_detector.api.app:app --reload
   ```
   Then open `http://127.0.0.1:8000/` for the browser demo.

   If `models/checkpoints/baseline.pt` is missing, the UI still starts in demo mode so you can show the upload flow. For meaningful predictions, place a trained checkpoint at that path first.

## Next Steps (recommended)
- Replace placeholder face detection fallback with your chosen detector (MTCNN/RetinaFace if needed).
- Add temporal augmentations and balanced sampling.
- Add evaluation scripts (AUC, F1, precision/recall).
- Add experiment tracking (Weights & Biases / MLflow).
- Harden API for production usage (auth, queueing, model warm-loading, monitoring).
