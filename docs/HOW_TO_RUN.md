# How To Run Deepfake Detector (From Scratch)

This guide covers everything from cloning the repository to starting the web app server.

## 1) Clone the repository

```bash
git clone https://github.com/neosac04/deepfake-detector.git
cd deepfake-detector
```

## 2) Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4) (Optional) Add a trained checkpoint

Place your trained model file at:

```text
models/checkpoints/baseline.pt
```

Notes:

- If this file exists, predictions use your trained model.
- If it is missing, the app still starts in demo mode (good for presentation flow), but predictions are not meaningful.

## 5) Start the app server

From the project root:

```bash
PYTHONPATH=src python -m uvicorn deepfake_detector.api.app:app --host 127.0.0.1 --port 8000
```

If you want auto-reload during development:

```bash
PYTHONPATH=src python -m uvicorn deepfake_detector.api.app:app --reload
```

## 6) Open the web demo

Open in your browser:

```text
http://127.0.0.1:8000/
```

Use the upload box to submit an image or video and view `FAKE`/`REAL` prediction and confidence.

## 7) Quick verification

In another terminal, verify server health:

```bash
curl -sS http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## 8) Train TensorFlow REAL/FAKE image model (optional)

Expected dataset layout:

```text
dataset/
  train/
    real/
    fake/
  validation/
    real/
    fake/
```

Train and fine-tune MobileNetV2 model:

```bash
python scripts/train_image_tf.py --mode train --dataset-dir ~/Downloads/archive/dataset
```

This saves:

- Model file at `models/exported/mobilenetv2_real_fake.keras`
- Class index map at `models/exported/mobilenetv2_real_fake.classes.json`
- Evaluation plots/reports under `models/exported/tf_eval/`

Run inference on one image without retraining:

```bash
python scripts/train_image_tf.py --mode infer --image-path path/to/image.jpg --model-path models/exported/mobilenetv2_real_fake.keras
```

## Common issues

1. `No module named uvicorn`

Install dependencies again:

```bash
pip install -r requirements.txt
```

1. `address already in use` on port 8000

Either stop the process using port 8000 or run on another port:

```bash
PYTHONPATH=src python -m uvicorn deepfake_detector.api.app:app --host 127.0.0.1 --port 8001
```

1. `Model is not loaded yet`

Make sure:

- `configs/default.yaml` exists
- Your checkpoint is at `models/checkpoints/baseline.pt` (if you want real predictions)
