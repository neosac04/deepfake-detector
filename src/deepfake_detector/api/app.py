from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

try:
    import numpy as np
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.models import load_model as tf_load_model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
except ImportError:  # pragma: no cover
    np = None
    preprocess_input = None
    tf_load_model = None
    img_to_array = None
    load_img = None

from deepfake_detector.config import load_config
from deepfake_detector.models.resnext_lstm import ResNeXtLSTM
from deepfake_detector.pipelines.inference_pipeline import load_checkpoint, predict_image, predict_video

app = FastAPI(title="Deepfake Detector API")

CFG_PATH = Path("configs/default.yaml")
CHECKPOINT_PATH = Path("models/checkpoints/baseline.pt")
TF_IMAGE_MODEL_PATH = Path("models/exported/mobilenetv2_real_fake.keras")
TF_IMAGE_CLASSES_PATH = Path("models/exported/mobilenetv2_real_fake.classes.json")
TF_IMAGE_THRESHOLD_PATH = Path("models/exported/mobilenetv2_real_fake.threshold.json")
MODEL = None
LABELS = {0: "FAKE", 1: "REAL"}
MODEL_SOURCE = "unavailable"
TF_IMAGE_MODEL = None
TF_CLASS_INDICES = {"fake": 0, "real": 1}
TF_IMAGE_MODEL_SOURCE = "unavailable"
TF_REAL_THRESHOLD = 0.5
REAL_THRESHOLD = 0.3
FAKE_THRESHOLD = 0.7
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _build_model(cfg):
    return ResNeXtLSTM(
        num_classes=cfg.model.num_classes,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        pretrained_backbone=cfg.model.pretrained_backbone,
    )


def _predict_image_tf(image_path: Path, image_size: int) -> tuple[int, float]:
    if TF_IMAGE_MODEL is None:
        raise ValueError("TensorFlow image model is not loaded.")

    # Prefer the TensorFlow model's declared input size to avoid shape mismatches.
    target_size = image_size
    try:
        input_shape = TF_IMAGE_MODEL.input_shape
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3:
            model_h = input_shape[1]
            model_w = input_shape[2]
            if isinstance(model_h, int) and isinstance(model_w, int) and model_h > 0 and model_w > 0:
                target_size = int(model_h)
    except Exception:
        pass

    image = load_img(image_path, target_size=(target_size, target_size))
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    raw_score = float(TF_IMAGE_MODEL.predict(arr, verbose=0)[0][0])
    real_idx = int(TF_CLASS_INDICES.get("real", 1))
    prob_real = raw_score if real_idx == 1 else 1.0 - raw_score
    pred_idx = 1 if prob_real >= TF_REAL_THRESHOLD else 0
    confidence = prob_real if pred_idx == 1 else 1.0 - prob_real
    return pred_idx, confidence


def _infer_media(tmp_path: Path, cfg):
    suffix = tmp_path.suffix.lower()
    is_image = suffix in IMAGE_EXTENSIONS
    if is_image:
        return predict_image(
            model=MODEL,
            image_path=tmp_path,
            sequence_length=cfg.data.sequence_length,
            image_size=cfg.data.image_size,
            device=cfg.inference.device,
        )
    return predict_video(
        model=MODEL,
        video_path=tmp_path,
        sequence_length=cfg.data.sequence_length,
        image_size=cfg.data.image_size,
        device=cfg.inference.device,
    )


def _prob_real_from_prediction(pred_idx: int, confidence: float) -> float:
    clipped = min(1.0, max(0.0, float(confidence)))
    return clipped if int(pred_idx) == 1 else 1.0 - clipped


@app.on_event("startup")
def startup_event() -> None:
    global MODEL, LABELS, MODEL_SOURCE, TF_IMAGE_MODEL, TF_CLASS_INDICES, TF_IMAGE_MODEL_SOURCE, TF_REAL_THRESHOLD
    if not CFG_PATH.exists():
        return
    cfg = load_config(CFG_PATH)
    LABELS = cfg.labels
    model = _build_model(cfg)
    model = model.to(cfg.inference.device)
    model.eval()
    if CHECKPOINT_PATH.exists():
        MODEL = load_checkpoint(model, CHECKPOINT_PATH, cfg.inference.device)
        MODEL_SOURCE = "checkpoint"
    else:
        MODEL = model
        MODEL_SOURCE = "demo"

    if tf_load_model is not None and TF_IMAGE_MODEL_PATH.exists():
        TF_IMAGE_MODEL = tf_load_model(TF_IMAGE_MODEL_PATH)
        TF_IMAGE_MODEL_SOURCE = "checkpoint"
        TF_REAL_THRESHOLD = 0.5
        if TF_IMAGE_CLASSES_PATH.exists():
            try:
                loaded = json.loads(TF_IMAGE_CLASSES_PATH.read_text(encoding="utf-8"))
                TF_CLASS_INDICES = {str(k).lower(): int(v) for k, v in loaded.items()}
            except Exception:
                TF_CLASS_INDICES = {"fake": 0, "real": 1}
        if TF_IMAGE_THRESHOLD_PATH.exists():
            try:
                threshold_payload = json.loads(TF_IMAGE_THRESHOLD_PATH.read_text(encoding="utf-8"))
                TF_REAL_THRESHOLD = float(threshold_payload.get("real_threshold", 0.5))
            except Exception:
                TF_REAL_THRESHOLD = 0.5
    else:
        TF_IMAGE_MODEL = None
        TF_IMAGE_MODEL_SOURCE = "unavailable"
        TF_REAL_THRESHOLD = 0.5


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Deepfake Detector Demo</title>
    <style>
        :root {{
            color-scheme: dark;
            --bg: #eef1f4;
            --panel: rgba(255, 255, 255, 0.9);
            --panel-strong: #ffffff;
            --panel-border: rgba(15, 23, 42, 0.08);
            --text: #18212f;
            --muted: #5f6b7a;
            --accent: #2f6fed;
            --accent-soft: rgba(47, 111, 237, 0.12);
            --good: #166534;
            --warn: #a16207;
            --bad: #b42318;
            --shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            min-height: 100vh;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background:
                radial-gradient(circle at top left, rgba(47, 111, 237, 0.08), transparent 30%),
                linear-gradient(180deg, #f6f8fb 0%, #edf1f5 100%);
            color: var(--text);
            display: grid;
            place-items: center;
            padding: 28px 16px;
        }}
        .shell {{
            width: min(1080px, 100%);
            display: grid;
            gap: 18px;
            grid-template-columns: 1fr 0.9fr;
        }}
        .hero, .card {{
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
        }}
        .hero {{ padding: 28px; }}
        .card {{ padding: 22px; }}
        h1 {{
            margin: 0 0 10px;
            font-size: clamp(1.8rem, 4vw, 2.6rem);
            line-height: 1.05;
            letter-spacing: -0.03em;
            max-width: 12ch;
        }}
        p {{ color: var(--muted); line-height: 1.6; margin: 0; }}
        .intro {{ margin-bottom: 18px; }}
        .steps {{
            margin: 18px 0 0;
            padding: 0;
            list-style: none;
            display: grid;
            gap: 10px;
        }}
        .steps li {{
            display: flex;
            gap: 10px;
            align-items: flex-start;
            color: var(--text);
            line-height: 1.5;
            padding: 12px 14px;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.65);
            border: 1px solid rgba(15, 23, 42, 0.06);
        }}
        .steps span {{
            flex: 0 0 auto;
            width: 22px;
            height: 22px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            display: inline-grid;
            place-items: center;
            font-size: 0.8rem;
            font-weight: 700;
            margin-top: 2px;
        }}
        .upload {{ display: grid; gap: 14px; }}
        label {{ font-size: 0.95rem; color: var(--muted); }}
        input[type="file"] {{
            width: 100%;
            padding: 14px;
            border-radius: 14px;
            background: var(--panel-strong);
            color: var(--text);
            border: 1px solid rgba(15, 23, 42, 0.12);
        }}
        .button {{
            appearance: none;
            border: 0;
            padding: 14px 18px;
            border-radius: 14px;
            background: var(--accent);
            color: #ffffff;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
        }}
        .button:disabled {{ opacity: 0.6; cursor: wait; }}
        .preview, .result {{
            margin-top: 16px;
            border-radius: 16px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid rgba(15, 23, 42, 0.08);
        }}
        .preview {{ min-height: 240px; display: grid; place-items: center; }}
        .preview img, .preview video {{ width: 100%; display: block; max-height: 420px; object-fit: contain; }}
        .preview .empty {{ padding: 28px; text-align: center; color: var(--muted); }}
        .result {{ padding: 18px; min-height: 92px; display: grid; align-content: center; }}
        .result.empty {{ color: var(--muted); }}
        .result-state {{ display: grid; gap: 4px; }}
        .result-state.real {{ color: var(--good); }}
        .result-state.fake {{ color: var(--bad); }}
        .result-state.uncertain {{ color: var(--warn); }}
        .result-label {{
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        .result-confidence {{
            font-size: 1rem;
            color: var(--muted);
        }}
        .error {{ color: #fecaca; }}
        @media (max-width: 920px) {{
            .shell {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <h1>Check if an image is real or fake</h1>
            <p class="intro">Upload a clear image or video and the demo will return a simple prediction with confidence.</p>
            <ul class="steps">
                <li><span>1</span><div>Choose a file from your device.</div></li>
                <li><span>2</span><div>Click Analyze media.</div></li>
                <li><span>3</span><div>Review the final result on the right.</div></li>
            </ul>
        </section>

        <section class="card">
            <form id="upload-form" class="upload">
                <label for="file-input">Upload an image or video</label>
                <input id="file-input" name="file" type="file" accept="image/*,video/*,.jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff,.mp4,.mov,.avi,.mkv" required />
                <button class="button" id="submit-button" type="submit">Analyze media</button>
            </form>
            <div class="preview" id="preview">
                <div class="empty">Your preview will appear here.</div>
            </div>
            <div class="result empty" id="result">
                <div>Prediction will appear here after upload.</div>
            </div>
        </section>
    </main>
    <script>
        const form = document.getElementById('upload-form');
        const fileInput = document.getElementById('file-input');
        const preview = document.getElementById('preview');
        const result = document.getElementById('result');
        const submitButton = document.getElementById('submit-button');
        let objectUrl = null;

        function setResult(html, isError = false) {{
            result.innerHTML = html;
            result.classList.toggle('error', isError);
            result.classList.remove('empty');
        }}

        function clearPreview() {{
            if (objectUrl) {{
                URL.revokeObjectURL(objectUrl);
                objectUrl = null;
            }}
        }}

        fileInput.addEventListener('change', () => {{
            const file = fileInput.files && fileInput.files[0];
            clearPreview();
            if (!file) {{
                preview.innerHTML = '<div class="empty">Your preview will appear here.</div>';
                result.className = 'result empty';
                result.innerHTML = '<div>Prediction will appear here after upload.</div>';
                return;
            }}

            objectUrl = URL.createObjectURL(file);
            const isImage = file.type.startsWith('image/');
            const isVideo = file.type.startsWith('video/');
            preview.innerHTML = isImage
                ? `<img src="${{objectUrl}}" alt="Selected image preview" />`
                : isVideo
                    ? `<video src="${{objectUrl}}" controls muted playsinline></video>`
                    : '<div class="empty">Unsupported preview type, but the file can still be uploaded.</div>';
        }});

        form.addEventListener('submit', async (event) => {{
            event.preventDefault();
            const file = fileInput.files && fileInput.files[0];
            if (!file) {{
                setResult('<div class="meta">Choose a file first.</div>', true);
                return;
            }}

            submitButton.disabled = true;
            setResult('<div class="result-state uncertain"><div class="result-label">Analyzing...</div><div class="result-confidence">Please wait.</div></div>');

            try {{
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/predict', {{
                    method: 'POST',
                    body: formData,
                }});

                const payload = await response.json();
                if (!response.ok) {{
                    throw new Error(payload.detail || 'Prediction failed');
                }}

                const confidencePct = (Number(payload.confidence) * 100).toFixed(1);
                const resultLabel = payload.prediction_label === 'REAL'
                    ? 'Real'
                    : payload.prediction_label === 'FAKE'
                        ? 'Fake'
                        : 'Uncertain';
                const stateClass = resultLabel.toLowerCase();

                setResult(`
                    <div class="result-state ${{stateClass}}">
                        <div class="result-label">Result: ${{resultLabel}}</div>
                        <div class="result-confidence">Confidence: ${{confidencePct}}%</div>
                    </div>
                `);
            }} catch (error) {{
                setResult(`<div>${{error.message}}</div>`, true);
            }} finally {{
                submitButton.disabled = false;
            }}
        }});
    </script>
</body>
</html>"""


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    cfg = load_config(CFG_PATH)
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        is_image = suffix.lower() in IMAGE_EXTENSIONS or (file.content_type or "").startswith("image/")
        if is_image:
            if TF_IMAGE_MODEL is not None and preprocess_input is not None and img_to_array is not None and load_img is not None and np is not None:
                try:
                    pred_idx, confidence = _predict_image_tf(
                        image_path=tmp_path,
                        image_size=cfg.data.image_size,
                    )
                except Exception:
                    pred_idx, confidence = predict_image(
                        model=MODEL,
                        image_path=tmp_path,
                        sequence_length=cfg.data.sequence_length,
                        image_size=cfg.data.image_size,
                        device=cfg.inference.device,
                    )
            else:
                pred_idx, confidence = predict_image(
                    model=MODEL,
                    image_path=tmp_path,
                    sequence_length=cfg.data.sequence_length,
                    image_size=cfg.data.image_size,
                    device=cfg.inference.device,
                )
        else:
            pred_idx, confidence = predict_video(
                model=MODEL,
                video_path=tmp_path,
                sequence_length=cfg.data.sequence_length,
                image_size=cfg.data.image_size,
                device=cfg.inference.device,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    prob_real = _prob_real_from_prediction(pred_idx, confidence)
    prob_fake = 1.0 - prob_real
    top_confidence = max(prob_real, prob_fake)

    if prob_fake > FAKE_THRESHOLD:
        prediction_index = 0
        prediction_label = LABELS.get(0, "FAKE")
        is_uncertain = False
        uncertainty_margin = prob_fake - FAKE_THRESHOLD
    elif prob_fake < REAL_THRESHOLD:
        prediction_index = 1
        prediction_label = LABELS.get(1, "REAL")
        is_uncertain = False
        uncertainty_margin = REAL_THRESHOLD - prob_fake
    else:
        prediction_index = -1
        prediction_label = "UNCERTAIN"
        is_uncertain = True
        uncertainty_margin = min(prob_fake - REAL_THRESHOLD, FAKE_THRESHOLD - prob_fake)

    return {
        "prediction_index": prediction_index,
        "prediction_label": prediction_label,
        "confidence": float(top_confidence),
        "prob_real": float(prob_real),
        "prob_fake": float(prob_fake),
        "real_threshold": float(REAL_THRESHOLD),
        "fake_threshold": float(FAKE_THRESHOLD),
        "uncertainty_margin": float(max(0.0, uncertainty_margin)),
        "is_uncertain": bool(is_uncertain),
    }
