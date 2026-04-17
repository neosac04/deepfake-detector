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

from backend.config import load_config
from backend.models.resnext_lstm import ResNeXtLSTM
from backend.utilities.pipelines.inference_pipeline import (
    load_checkpoint,
    predict_image,
    predict_video,
)

app = FastAPI(title="Deepfake Detector API")

CFG_PATH = Path("backend/config/default.yaml")
CHECKPOINT_PATH = Path("models/mobilenetv2/baseline.pt")
TF_IMAGE_MODEL_PATH = Path("models/mobilenetv2/mobilenetv2.h5")
TF_IMAGE_CLASSES_PATH = Path("models/mobilenetv2/mobilenetv2.classes.json")
TF_IMAGE_THRESHOLD_PATH = Path("models/mobilenetv2/mobilenetv2.threshold.json")
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

# Multi-model registry for TensorFlow image models
MODEL_REGISTRY = {}  # {model_name: {"model": model, "classes": classes_dict, "threshold": threshold, "metrics": metrics}}
DEFAULT_TF_MODEL = "mobilenetv2"
MODELS_DIR = Path("models")


def _build_model(cfg):
    return ResNeXtLSTM(
        num_classes=cfg.model.num_classes,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        pretrained_backbone=cfg.model.pretrained_backbone,
    )


def _load_tf_model(model_name: str) -> dict:
    """Load a TensorFlow model with its metadata.
    
    Args:
        model_name: Name of the model directory (e.g., 'mobilenetv2', 'efficientnet')
    
    Returns:
        Dict with keys: model, classes, threshold, metrics, name
    """
    if tf_load_model is None:
        raise RuntimeError("TensorFlow not available")
    
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")
    
    # Find .keras file
    keras_files = list(model_dir.glob("*.keras"))
    if not keras_files:
        raise ValueError(f"No .keras model file found in {model_dir}")
    
    model_path = keras_files[0]
    
    # Find metadata files - try multiple naming patterns
    classes_path = None
    threshold_path = None
    metrics_path = None
    
    # Try both naming patterns for each metadata file
    candidates_classes = [
        model_dir / f"{model_name}_deepfake.classes.json",
        model_dir / f"{model_name}_real_fake.classes.json",
    ]
    candidates_threshold = [
        model_dir / f"{model_name}_deepfake.threshold.json",
        model_dir / f"{model_name}_real_fake.threshold.json",
    ]
    candidates_metrics = [
        model_dir / f"{model_name}_deepfake_metrics.json",
        model_dir / f"{model_name}_deepfake.metrics.json",
        model_dir / f"{model_name}_real_fake_metrics.json",
        model_dir / f"{model_name}_real_fake.metrics.json",
    ]
    
    for candidate in candidates_classes:
        if candidate.exists():
            classes_path = candidate
            break
    
    for candidate in candidates_threshold:
        if candidate.exists():
            threshold_path = candidate
            break
    
    for candidate in candidates_metrics:
        if candidate.exists():
            metrics_path = candidate
            break
    
    # Load model
    loaded_model = tf_load_model(str(model_path))
    
    # Load classes
    classes = {"Fake": 0, "Real": 1}
    if classes_path and classes_path.exists():
        try:
            loaded = json.loads(classes_path.read_text(encoding="utf-8"))
            classes = {str(k): int(v) for k, v in loaded.items()}
        except Exception:
            pass
    
    # Load threshold
    threshold = 0.5
    if threshold_path and threshold_path.exists():
        try:
            threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
            threshold = float(threshold_payload.get("real_threshold", 0.5))
        except Exception:
            pass
    
    # Load metrics
    metrics = {}
    if metrics_path and metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    return {
        "model": loaded_model,
        "classes": classes,
        "threshold": threshold,
        "metrics": metrics,
        "name": model_name,
    }


def _discover_tf_models() -> list[str]:
    """Discover available TensorFlow models in models/ directory."""
    if not MODELS_DIR.exists():
        return []
    
    available_models = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir() and list(model_dir.glob("*.keras")):
            available_models.append(model_dir.name)
    
    return sorted(available_models)


def _predict_image_tf(image_path: Path, image_size: int, model_name: str = None) -> tuple[int, float]:
    """Make prediction on image using TensorFlow model.
    
    Args:
        image_path: Path to image file
        image_size: Target image size
        model_name: Name of model to use (defaults to DEFAULT_TF_MODEL)
    
    Returns:
        Tuple of (prediction_idx, confidence)
    """
    if model_name is None:
        model_name = DEFAULT_TF_MODEL
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model not available: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_info = MODEL_REGISTRY[model_name]
    tf_model = model_info["model"]
    classes = model_info["classes"]
    threshold = model_info["threshold"]
    
    if tf_model is None:
        raise ValueError(f"TensorFlow model not loaded: {model_name}")

    # Prefer the TensorFlow model's declared input size to avoid shape mismatches.
    target_size = image_size
    try:
        input_shape = tf_model.input_shape
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

    raw_score = float(tf_model.predict(arr, verbose=0)[0][0])
    real_idx = int(classes.get("Real", 1))
    prob_real = raw_score if real_idx == 1 else 1.0 - raw_score
    pred_idx = 1 if prob_real >= threshold else 0
    confidence = prob_real if pred_idx == 1 else 1.0 - prob_real
    return pred_idx, confidence


def _predict_image_ensemble(image_path: Path, image_size: int) -> dict:
    """Make ensemble predictions using all available TensorFlow models.
    
    Returns ensemble prediction with individual model predictions.
    
    Returns:
        Dict with keys: 
            - prediction: Final ensemble prediction (0=Fake, 1=Real)
            - confidence: Final ensemble confidence [0,1]
            - label: Final ensemble label ("Real" or "Fake")
            - models: List of individual model predictions
    """
    if not MODEL_REGISTRY:
        raise ValueError("No models available for ensemble prediction")
    
    model_predictions = []
    real_scores = []  # Confidence scores for "Real" class
    predictions = []  # Individual predictions (0/1)
    
    # Run all available models
    for model_name in sorted(MODEL_REGISTRY.keys()):
        try:
            pred_idx, confidence = _predict_image_tf(
                image_path=image_path,
                image_size=image_size,
                model_name=model_name,
            )
            
            # Convert to probability of Real (1)
            prob_real = confidence if pred_idx == 1 else 1.0 - confidence
            real_scores.append(prob_real)
            predictions.append(pred_idx)
            
            label = "Real" if pred_idx == 1 else "Fake"
            model_predictions.append({
                "model": model_name,
                "prediction": label,
                "confidence": round(float(confidence), 4),
                "threshold": MODEL_REGISTRY[model_name]["threshold"],
            })
        except Exception as e:
            print(f"Error running model {model_name}: {e}")
            continue
    
    if not model_predictions:
        raise ValueError("All models failed to produce predictions")
    
    # Ensemble decision logic
    # 1. Average the "Real" class probabilities
    avg_prob_real = sum(real_scores) / len(real_scores) if real_scores else 0.5
    
    # 2. Voting - count predictions
    num_real = sum(predictions)
    num_models = len(predictions)
    num_fake = num_models - num_real
    
    # 3. Determine final prediction
    # Use voting first, then average confidence for tie-breaking
    if num_real > num_fake:
        final_pred = 1  # Real
        final_confidence = avg_prob_real
    elif num_fake > num_real:
        final_pred = 0  # Fake
        final_confidence = 1.0 - avg_prob_real
    else:
        # Tie: use average probability
        final_pred = 1 if avg_prob_real >= 0.5 else 0
        final_confidence = max(avg_prob_real, 1.0 - avg_prob_real)
    
    final_label = "Real" if final_pred == 1 else "Fake"
    
    return {
        "prediction": final_pred,
        "confidence": round(float(final_confidence), 4),
        "label": final_label,
        "models": model_predictions,
        "ensemble_method": "voting + averaged confidence",
    }


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
    global MODEL, LABELS, MODEL_SOURCE, TF_IMAGE_MODEL, TF_CLASS_INDICES, TF_IMAGE_MODEL_SOURCE, TF_REAL_THRESHOLD, MODEL_REGISTRY
    
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

    # Load all available TensorFlow image models
    if tf_load_model is not None:
        available_models = _discover_tf_models()
        for model_name in available_models:
            try:
                model_info = _load_tf_model(model_name)
                MODEL_REGISTRY[model_name] = model_info
                print(f"Loaded TF model: {model_name}")
            except Exception as e:
                print(f"Failed to load TF model {model_name}: {e}")
        
        # Backward compatibility: set default model to global variables
        if DEFAULT_TF_MODEL in MODEL_REGISTRY:
            model_info = MODEL_REGISTRY[DEFAULT_TF_MODEL]
            TF_IMAGE_MODEL = model_info["model"]
            TF_IMAGE_MODEL_SOURCE = "checkpoint"
            TF_REAL_THRESHOLD = model_info["threshold"]
            TF_CLASS_INDICES = {str(k).lower(): int(v) for k, v in model_info["classes"].items()}
        else:
            TF_IMAGE_MODEL = None
            TF_IMAGE_MODEL_SOURCE = "unavailable"
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
    <title>Deepfake Detector</title>
    <style>
        :root {{
            --bg: #eef1f4;
            --panel: rgba(255, 255, 255, 0.9);
            --panel-strong: #ffffff;
            --panel-border: rgba(15, 23, 42, 0.08);
            --text: #18212f;
            --muted: #5f6b7a;
            --accent: #2f6fed;
            --accent-soft: rgba(47, 111, 237, 0.12);
            --good: #166534;
            --bad: #b42318;
            --warn: #a16207;
            --shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            min-height: 100vh;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: linear-gradient(180deg, #f6f8fb 0%, #edf1f5 100%);
            color: var(--text);
            padding: 20px 16px;
        }}
        header {{
            max-width: 1400px;
            margin: 0 auto 40px;
            text-align: center;
        }}
        h1 {{
            margin: 0 0 8px;
            font-size: 2.2rem;
            letter-spacing: -0.02em;
        }}
        .subtitle {{
            color: var(--muted);
            font-size: 0.95rem;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .windows {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }}
        .window {{
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            height: 600px;
        }}
        .window-header {{
            padding: 20px;
            border-bottom: 1px solid var(--panel-border);
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .window-title {{
            font-size: 1.1rem;
            font-weight: 700;
            margin: 0;
            flex: 1;
        }}
        .step-badge {{
            background: var(--accent-soft);
            color: var(--accent);
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            font-weight: 700;
            font-size: 0.9rem;
        }}
        .window-content {{
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .upload-area {{
            display: grid;
            gap: 14px;
        }}
        .file-input-label {{
            display: block;
            font-size: 0.9rem;
            color: var(--muted);
            margin-bottom: 8px;
        }}
        input[type="file"] {{
            width: 100%;
            padding: 12px;
            border-radius: 10px;
            background: var(--panel-strong);
            color: var(--text);
            border: 1px solid var(--panel-border);
            cursor: pointer;
        }}
        .preview {{
            border-radius: 12px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.04);
            border: 1px solid var(--panel-border);
            min-height: 250px;
            display: grid;
            place-items: center;
        }}
        .preview img, .preview video {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        .preview .empty {{
            padding: 40px 20px;
            text-align: center;
            color: var(--muted);
        }}
        .button {{
            appearance: none;
            border: 0;
            padding: 12px 20px;
            border-radius: 10px;
            background: var(--accent);
            color: white;
            font-weight: 700;
            font-size: 0.95rem;
            cursor: pointer;
            transition: opacity 0.2s;
        }}
        .button:hover:not(:disabled) {{
            opacity: 0.9;
        }}
        .button:disabled {{
            opacity: 0.5;
            cursor: wait;
        }}
        .result-main {{
            display: grid;
            gap: 16px;
        }}
        .prediction-card {{
            background: rgba(15, 23, 42, 0.04);
            border: 2px solid var(--panel-border);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .prediction-card.real {{
            border-color: rgba(22, 101, 52, 0.3);
            background: rgba(22, 101, 52, 0.05);
        }}
        .prediction-card.fake {{
            border-color: rgba(180, 35, 24, 0.3);
            background: rgba(180, 35, 24, 0.05);
        }}
        .prediction-label {{
            font-size: 1.3rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }}
        .prediction-label.real {{
            color: var(--good);
        }}
        .prediction-label.fake {{
            color: var(--bad);
        }}
        .prediction-confidence {{
            font-size: 0.9rem;
            color: var(--muted);
            margin-top: 4px;
        }}
        .models-breakdown {{
            display: grid;
            gap: 12px;
        }}
        .model-item {{
            background: rgba(15, 23, 42, 0.04);
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            padding: 12px;
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            gap: 12px;
        }}
        .model-name {{
            font-weight: 600;
            font-size: 0.9rem;
        }}
        .model-prediction {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
        }}
        .model-badge {{
            background: var(--accent-soft);
            color: var(--accent);
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
        }}
        .model-badge.real {{
            background: rgba(22, 101, 52, 0.15);
            color: var(--good);
        }}
        .model-badge.fake {{
            background: rgba(180, 35, 24, 0.15);
            color: var(--bad);
        }}
        .empty-state {{
            color: var(--muted);
            text-align: center;
            padding: 40px 20px;
            display: grid;
            place-items: center;
            height: 100%;
        }}
        .error-msg {{
            background: rgba(180, 35, 24, 0.1);
            border: 1px solid rgba(180, 35, 24, 0.3);
            color: var(--bad);
            padding: 12px;
            border-radius: 8px;
            font-size: 0.9rem;
        }}
        .nav-buttons {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .nav-button {{
            appearance: none;
            border: 1px solid var(--panel-border);
            padding: 12px 20px;
            border-radius: 10px;
            background: var(--panel);
            color: var(--text);
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .nav-button:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}
        .nav-button.active {{
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }}
        .loading {{
            display: grid;
            place-items: center;
            height: 100%;
        }}
        .spinner {{
            width: 32px;
            height: 32px;
            border: 3px solid var(--panel-border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        @media (max-width: 1024px) {{
            .windows {{
                grid-template-columns: 1fr;
                height: auto;
            }}
            .window {{
                height: auto;
                min-height: 400px;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>Deepfake Detector</h1>
        <p class="subtitle">Analyze images to detect authentic vs deepfake content</p>
    </header>

    <div class="container">
        <div class="windows">
            <!-- Window 1: Upload -->
            <div class="window">
                <div class="window-header">
                    <div class="step-badge">1</div>
                    <h2 class="window-title">Upload Image</h2>
                </div>
                <div class="window-content">
                    <div class="upload-area">
                        <label class="file-input-label">Select an image or video</label>
                        <input id="file-input" type="file" accept="image/*,video/*,.jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff,.mp4,.mov,.avi,.mkv" />
                        <button class="button" id="analyze-btn">Analyze</button>
                    </div>
                    <div class="preview" id="preview">
                        <div class="empty">Preview will appear here</div>
                    </div>
                </div>
            </div>

            <!-- Window 2: Results -->
            <div class="window">
                <div class="window-header">
                    <div class="step-badge">2</div>
                    <h2 class="window-title">Analysis Results</h2>
                </div>
                <div class="window-content" id="results-content">
                    <div class="empty-state">Upload an image to see results</div>
                </div>
            </div>

            <!-- Window 3: Explanations -->
            <div class="window">
                <div class="window-header">
                    <div class="step-badge">3</div>
                    <h2 class="window-title">Model Details</h2>
                </div>
                <div class="window-content" id="explanations-content">
                    <div class="empty-state">Details will appear after analysis</div>
                </div>
            </div>
        </div>

        <div class="nav-buttons">
            <button class="nav-button active" onclick="scrollToWindow(0)">1. Upload</button>
            <button class="nav-button" onclick="scrollToWindow(1)">2. Results</button>
            <button class="nav-button" onclick="scrollToWindow(2)">3. Details</button>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('file-input');
        const analyzeBtn = document.getElementById('analyze-btn');
        const previewDiv = document.getElementById('preview');
        const resultsDiv = document.getElementById('results-content');
        const explanationsDiv = document.getElementById('explanations-content');
        let objectUrl = null;
        let currentData = null;

        function scrollToWindow(index) {{
            const windows = document.querySelectorAll('.window');
            const navButtons = document.querySelectorAll('.nav-button');
            if (windows[index]) {{
                windows[index].scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                navButtons.forEach((btn, i) => btn.classList.toggle('active', i === index));
            }}
        }}

        fileInput.addEventListener('change', () => {{
            const file = fileInput.files?.[0];
            if (!file) {{
                previewDiv.innerHTML = '<div class="empty">Preview will appear here</div>';
                resultsDiv.innerHTML = '<div class="empty-state">Upload an image to see results</div>';
                return;
            }}

            if (objectUrl) URL.revokeObjectURL(objectUrl);
            objectUrl = URL.createObjectURL(file);
            const isImage = file.type.startsWith('image/');
            const isVideo = file.type.startsWith('video/');
            
            if (isImage) {{
                previewDiv.innerHTML = `<img src="${{objectUrl}}" alt="Preview" />`;
            }} else if (isVideo) {{
                previewDiv.innerHTML = `<video src="${{objectUrl}}" controls muted></video>`;
            }} else {{
                previewDiv.innerHTML = '<div class="empty">Unsupported file type</div>';
            }}
            
            analyzeBtn.textContent = 'Analyze';
            analyzeBtn.disabled = false;
        }});

        analyzeBtn.addEventListener('click', async () => {{
            const file = fileInput.files?.[0];
            if (!file) {{
                resultsDiv.innerHTML = '<div class="error-msg">Please select a file first</div>';
                return;
            }}

            analyzeBtn.disabled = true;
            analyzeBtn.textContent = 'Analyzing...';
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

            try {{
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch('/predict', {{
                    method: 'POST',
                    body: formData,
                }});

                const data = await response.json();
                if (!response.ok) {{
                    throw new Error(data.detail || 'Analysis failed');
                }}

                currentData = data;
                displayResults(data);
                displayExplanations(data);
                scrollToWindow(1);
            }} catch (error) {{
                resultsDiv.innerHTML = `<div class="error-msg">${{error.message}}</div>`;
            }} finally {{
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze';
            }}
        }});

        function displayResults(data) {{
            const confidence = (Number(data.confidence) * 100).toFixed(1);
            const label = data.prediction_label === 'REAL' ? 'Real' : 'FAKE' ? 'Fake' : 'Uncertain';
            const stateClass = label.toLowerCase();

            let html = `
                <div class="result-main">
                    <div class="prediction-card ${{stateClass}}">
                        <div class="prediction-label ${{stateClass}}">${{label}}</div>
                        <div class="prediction-confidence">Confidence: ${{confidence}}%</div>
                    </div>
            `;

            if (data.model_predictions && data.model_predictions.length > 0) {{
                html += '<div class="models-breakdown"><div style="font-size: 0.9rem; font-weight: 600; color: var(--muted); margin-bottom: 4px;">Individual Model Predictions:</div>';
                for (const model of data.model_predictions) {{
                    const modelLabel = model.prediction.toLowerCase();
                    const modelConf = (Number(model.confidence) * 100).toFixed(1);
                    html += `
                        <div class="model-item">
                            <div class="model-name">${{model.model}}</div>
                            <div class="model-prediction">
                                <span class="model-badge ${{modelLabel}}">${{model.prediction}}</span>
                                <span style="color: var(--muted); font-size: 0.85rem;">${{modelConf}}%</span>
                            </div>
                        </div>
                    `;
                }}
                html += '</div>';
            }}

            html += '</div>';
            resultsDiv.innerHTML = html;
        }}

        function displayExplanations(data) {{
            let html = '<div style="display: grid; gap: 16px;">';
            
            if (data.model_predictions && data.model_predictions.length > 0) {{
                html += '<div>';
                html += '<h3 style="margin: 0 0 12px; font-size: 0.95rem; font-weight: 600;">Model Information</h3>';
                for (const model of data.model_predictions) {{
                    html += `
                        <div style="background: rgba(15, 23, 42, 0.04); padding: 12px; border-radius: 8px; margin-bottom: 8px;">
                            <div style="font-weight: 600; margin-bottom: 4px;">${{model.model}}</div>
                            <div style="font-size: 0.85rem; color: var(--muted);">
                                Threshold: ${{model.threshold}}<br/>
                                Prediction: ${{model.prediction}}<br/>
                                Confidence: ${{(Number(model.confidence) * 100).toFixed(1)}}%
                            </div>
                        </div>
                    `;
                }}
                html += '</div>';
            }}

            html += '<div>';
            html += '<h3 style="margin: 0 0 8px; font-size: 0.95rem; font-weight: 600;">Analysis Method</h3>';
            html += `<p style="margin: 0; font-size: 0.85rem; color: var(--muted); line-height: 1.5;">Ensemble predictions using voting and averaged confidence scores from multiple models.</p>`;
            html += '</div>';

            html += '</div>';
            explanationsDiv.innerHTML = html;
        }}
    </script>
</body>
</html>"""


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models/available")
def get_available_models() -> dict:
    """Return list of available TensorFlow image models with their metrics."""
    models_list = []
    for model_name in sorted(MODEL_REGISTRY.keys()):
        model_info = MODEL_REGISTRY[model_name]
        metrics = model_info.get("metrics", {})
        models_list.append({
            "name": model_name,
            "threshold": model_info.get("threshold", 0.5),
            "accuracy": metrics.get("accuracy"),
            "auc_roc": metrics.get("auc_roc"),
            "test_samples": metrics.get("test_samples"),
        })
    
    return {
        "models": models_list,
        "default_model": DEFAULT_TF_MODEL,
        "total_models": len(models_list),
    }


@app.get("/models/{model_name}/info")
def get_model_info(model_name: str) -> dict:
    """Return detailed information about a specific model."""
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_info = MODEL_REGISTRY[model_name]
    return {
        "name": model_name,
        "classes": model_info.get("classes", {}),
        "threshold": model_info.get("threshold", 0.5),
        "metrics": model_info.get("metrics", {}),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = None) -> dict:
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
            if MODEL_REGISTRY and preprocess_input is not None and img_to_array is not None and load_img is not None and np is not None:
                try:
                    # Use ensemble if no specific model requested, otherwise use single model
                    if model is None:
                        ensemble_result = _predict_image_ensemble(
                            image_path=tmp_path,
                            image_size=cfg.data.image_size,
                        )
                        pred_idx = ensemble_result["prediction"]
                        confidence = ensemble_result["confidence"]
                        individual_models = ensemble_result["models"]
                    else:
                        pred_idx, confidence = _predict_image_tf(
                            image_path=tmp_path,
                            image_size=cfg.data.image_size,
                            model_name=model,
                        )
                        individual_models = None
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Image prediction failed: {exc}") from exc
            else:
                pred_idx, confidence = predict_image(
                    model=MODEL,
                    image_path=tmp_path,
                    sequence_length=cfg.data.sequence_length,
                    image_size=cfg.data.image_size,
                    device=cfg.inference.device,
                )
                individual_models = None
        else:
            pred_idx, confidence = predict_video(
                model=MODEL,
                video_path=tmp_path,
                sequence_length=cfg.data.sequence_length,
                image_size=cfg.data.image_size,
                device=cfg.inference.device,
            )
            individual_models = None
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

    result = {
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
    
    # Add individual model predictions if ensemble was used
    if individual_models is not None:
        result["model_predictions"] = individual_models
        result["ensemble_method"] = "voting + averaged confidence"
    
    return result
