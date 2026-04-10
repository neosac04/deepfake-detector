from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

from deepfake_detector.config import load_config
from deepfake_detector.models.resnext_lstm import ResNeXtLSTM
from deepfake_detector.pipelines.inference_pipeline import load_checkpoint, predict_image, predict_video

app = FastAPI(title="Deepfake Detector API")

CFG_PATH = Path("configs/default.yaml")
CHECKPOINT_PATH = Path("models/checkpoints/baseline.pt")
MODEL = None
LABELS = {0: "FAKE", 1: "REAL"}
MODEL_SOURCE = "unavailable"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _build_model(cfg):
    return ResNeXtLSTM(
        num_classes=cfg.model.num_classes,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        pretrained_backbone=cfg.model.pretrained_backbone,
    )


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


@app.on_event("startup")
def startup_event() -> None:
    global MODEL, LABELS, MODEL_SOURCE
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    status_text = (
        "Loaded checkpoint" if MODEL_SOURCE == "checkpoint" else "Demo mode: using the configured model without a saved checkpoint"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Deepfake Detector Demo</title>
    <style>
        :root {{
            color-scheme: dark;
            --bg: #0b1020;
            --panel: rgba(12, 18, 35, 0.82);
            --panel-border: rgba(147, 197, 253, 0.18);
            --text: #eef2ff;
            --muted: #a5b4fc;
            --accent: #5eead4;
            --accent-2: #60a5fa;
            --danger: #fb7185;
            --shadow: 0 24px 80px rgba(2, 6, 23, 0.45);
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            min-height: 100vh;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.22), transparent 32%),
                radial-gradient(circle at right top, rgba(94, 234, 212, 0.14), transparent 28%),
                linear-gradient(160deg, #050816 0%, #0b1020 55%, #111827 100%);
            color: var(--text);
            display: grid;
            place-items: center;
            padding: 32px 16px;
        }}
        .shell {{
            width: min(1040px, 100%);
            display: grid;
            gap: 20px;
            grid-template-columns: 1.15fr 0.85fr;
        }}
        .hero, .card {{
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }}
        .hero {{ padding: 32px; }}
        .card {{ padding: 24px; }}
        .eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(94, 234, 212, 0.12);
            color: #bff9ee;
            font-size: 12px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 18px;
        }}
        h1 {{
            margin: 0 0 12px;
            font-size: clamp(2.2rem, 6vw, 4.8rem);
            line-height: 0.95;
            letter-spacing: -0.05em;
            max-width: 10ch;
        }}
        p {{ color: rgba(238, 242, 255, 0.8); line-height: 1.6; }}
        .stats {{ display: grid; gap: 12px; margin-top: 24px; grid-template-columns: repeat(3, minmax(0, 1fr)); }}
        .stat {{
            padding: 16px;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.12);
        }}
        .stat strong {{ display: block; font-size: 1.2rem; margin-bottom: 6px; }}
        .status {{
            margin: 20px 0 0;
            padding: 14px 16px;
            border-radius: 16px;
            background: rgba(96, 165, 250, 0.12);
            color: #dbeafe;
            border: 1px solid rgba(96, 165, 250, 0.24);
        }}
        .upload {{ display: grid; gap: 14px; }}
        label {{ font-size: 0.95rem; color: #cbd5e1; }}
        input[type="file"] {{
            width: 100%;
            padding: 14px;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.9);
            color: var(--text);
            border: 1px dashed rgba(147, 197, 253, 0.32);
        }}
        .button {{
            appearance: none;
            border: 0;
            padding: 14px 18px;
            border-radius: 16px;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: #03111b;
            font-weight: 800;
            font-size: 1rem;
            cursor: pointer;
        }}
        .button:disabled {{ opacity: 0.6; cursor: wait; }}
        .preview, .result {{
            margin-top: 16px;
            border-radius: 18px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.12);
        }}
        .preview {{ min-height: 240px; display: grid; place-items: center; }}
        .preview img, .preview video {{ width: 100%; display: block; max-height: 420px; object-fit: contain; }}
        .preview .empty {{ padding: 28px; text-align: center; color: rgba(226, 232, 240, 0.72); }}
        .result {{ padding: 18px; min-height: 92px; display: grid; gap: 6px; align-content: center; }}
        .result .label {{ font-size: 1.75rem; font-weight: 800; letter-spacing: -0.04em; }}
        .result .meta {{ color: rgba(226, 232, 240, 0.7); }}
        .error {{ color: #fecaca; }}
        @media (max-width: 920px) {{
            .shell {{ grid-template-columns: 1fr; }}
            .stats {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <div class="eyebrow">Deepfake detector demo</div>
            <h1>Upload media, get a real or fake prediction.</h1>
            <p>This presentation demo accepts both images and videos. It uses the same preprocessing and model inference pipeline as the API, so you can show a complete working flow in the browser.</p>
            <div class="stats">
                <div class="stat"><strong>Inputs</strong><span>JPG, PNG, MP4, MOV, and more</span></div>
                <div class="stat"><strong>Pipeline</strong><span>Frame sampling, face crop, classification</span></div>
                <div class="stat"><strong>Output</strong><span>FAKE or REAL with confidence</span></div>
            </div>
            <div class="status">Model status: {status_text}</div>
        </section>

        <section class="card">
            <form id="upload-form" class="upload">
                <label for="file-input">Choose an image or video</label>
                <input id="file-input" name="file" type="file" accept="image/*,video/*,.jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff,.mp4,.mov,.avi,.mkv" required />
                <button class="button" id="submit-button" type="submit">Analyze media</button>
            </form>
            <div class="preview" id="preview">
                <div class="empty">Your preview will appear here.</div>
            </div>
            <div class="result" id="result">
                <div class="meta">Prediction will appear here after upload.</div>
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
            setResult('<div class="meta">Analyzing media...</div>');

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

                const confidence = Number(payload.confidence).toFixed(4);
                setResult(`
                    <div class="meta">Prediction complete</div>
                    <div class="label">${{payload.prediction_label}}</div>
                    <div class="meta">Confidence: ${{confidence}}</div>
                `);
            }} catch (error) {{
                setResult(`<div class="meta">${{error.message}}</div>`, true);
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
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "prediction_index": int(pred_idx),
        "prediction_label": LABELS.get(int(pred_idx), str(pred_idx)),
        "confidence": float(confidence),
    }
