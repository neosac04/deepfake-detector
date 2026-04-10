from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix

try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required for this script. Install it with: pip install tensorflow"
    ) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def check_dataset_structure(dataset_dir: Path) -> None:
    required_paths = [
        dataset_dir,
        dataset_dir / "train",
        dataset_dir / "train" / "real",
        dataset_dir / "train" / "fake",
        dataset_dir / "validation",
        dataset_dir / "validation" / "real",
        dataset_dir / "validation" / "fake",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset structure is incomplete. Missing paths:\n" + "\n".join(missing)
        )


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def find_invalid_images(root_dir: Path) -> List[Path]:
    invalid_files: List[Path] = []
    for image_path in root_dir.rglob("*"):
        if not image_path.is_file() or not _is_image_file(image_path):
            continue
        try:
            with Image.open(image_path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError):
            invalid_files.append(image_path)
    return invalid_files


def summarize_dataset(dataset_dir: Path) -> Dict[str, int]:
    counts = {}
    for split in ["train", "validation"]:
        for label in ["real", "fake"]:
            folder = dataset_dir / split / label
            count = sum(1 for p in folder.rglob("*") if p.is_file() and _is_image_file(p))
            counts[f"{split}_{label}"] = count
    return counts


def load_data(
    dataset_dir: Path,
    image_size: int = 224,
    batch_size: int = 32,
):
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=12,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    # Separate generator for reproducible evaluation metrics.
    val_eval_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_gen, val_gen, val_eval_gen


def build_model(image_size: int = 224) -> Tuple[Model, Model]:
    base_model = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def fine_tune_model(model: Model, base_model: Model, unfreeze_top_n: int = 20) -> None:
    base_model.trainable = True

    # Keep most layers frozen and only unfreeze the top N layers.
    for layer in base_model.layers[:-unfreeze_top_n]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_top_n:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def merge_histories(initial_history, fine_tune_history):
    merged = {}
    for key in initial_history.history.keys():
        merged[key] = initial_history.history.get(key, []) + fine_tune_history.history.get(key, [])
    return merged


def plot_training_curves(history_dict: Dict[str, List[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history_dict.get("accuracy", [])) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict.get("accuracy", []), label="Train Accuracy")
    plt.plot(epochs, history_dict.get("val_accuracy", []), label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict.get("loss", []), label="Train Loss")
    plt.plot(epochs, history_dict.get("val_loss", []), label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    curves_path = output_dir / "training_curves.png"
    plt.tight_layout()
    plt.savefig(curves_path, dpi=150)
    plt.close()


def evaluate_model(model: Model, val_eval_gen, output_dir: Path, class_indices: Dict[str, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    val_eval_gen.reset()
    probs = model.predict(val_eval_gen, verbose=1)
    preds = (probs.ravel() >= 0.5).astype(int)
    y_true = val_eval_gen.classes

    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, target_names=class_names, digits=4)

    print("\nClassification Report:\n")
    print(report)

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    cm_path = output_dir / "confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()


def predict_single_image(
    model: Model,
    image_path: Path,
    image_size: int = 224,
    class_names: Tuple[str, str] = ("fake", "real"),
) -> Tuple[str, float]:
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = load_img(image_path, target_size=(image_size, image_size))
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Could not load image: {image_path}") from exc

    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    prob_real = float(model.predict(arr, verbose=0)[0][0])
    if prob_real >= 0.5:
        label = class_names[1]
        confidence = prob_real
    else:
        label = class_names[0]
        confidence = 1.0 - prob_real
    return label.upper(), confidence


def save_class_indices(class_indices: Dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(class_indices, indent=2), encoding="utf-8")


def train_pipeline(args) -> None:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    check_dataset_structure(dataset_dir)

    counts = summarize_dataset(dataset_dir)
    print("Dataset summary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    if min(counts.values()) == 0:
        raise ValueError("One or more dataset folders are empty. Please add images before training.")

    invalid_images = find_invalid_images(dataset_dir)
    if invalid_images:
        print("\nWarning: Found invalid image files. They may fail during training:")
        for path in invalid_images[:20]:
            print(f"  - {path}")
        if len(invalid_images) > 20:
            print(f"  ... and {len(invalid_images) - 20} more")

    train_gen, val_gen, val_eval_gen = load_data(
        dataset_dir=dataset_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    model, base_model = build_model(image_size=args.image_size)
    model.summary()

    print("\nStage 1: Training classification head...")
    history_initial = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        verbose=1,
    )

    print("\nStage 2: Fine-tuning top layers...")
    fine_tune_model(model, base_model, unfreeze_top_n=20)
    history_finetune = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.fine_tune_epochs,
        verbose=1,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"\nSaved trained model to: {model_path}")

    class_indices_path = model_path.with_suffix(".classes.json")
    save_class_indices(train_gen.class_indices, class_indices_path)
    print(f"Saved class indices to: {class_indices_path}")

    merged_history = merge_histories(history_initial, history_finetune)
    plot_training_curves(merged_history, output_dir=output_dir)
    evaluate_model(model, val_eval_gen, output_dir=output_dir, class_indices=train_gen.class_indices)


def infer_pipeline(args) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    image_path = Path(args.image_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path)

    class_names = ("fake", "real")
    class_indices_path = model_path.with_suffix(".classes.json")
    if class_indices_path.exists():
        try:
            class_indices = json.loads(class_indices_path.read_text(encoding="utf-8"))
            idx_to_name = {int(v): str(k) for k, v in class_indices.items()}
            if 0 in idx_to_name and 1 in idx_to_name:
                class_names = (idx_to_name[0], idx_to_name[1])
        except Exception:
            pass

    label, confidence = predict_single_image(
        model=model,
        image_path=image_path,
        image_size=args.image_size,
        class_names=class_names,
    )
    print(f"Prediction: {label} (confidence={confidence:.4f})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate a MobileNetV2 REAL-vs-FAKE image classifier with TensorFlow/Keras."
    )
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument(
        "--dataset-dir",
        default=str(Path.home() / "Downloads" / "archive" / "dataset"),
        help="Path to dataset root containing train/ and validation/ directories.",
    )
    parser.add_argument(
        "--model-path",
        default="models/exported/mobilenetv2_real_fake.keras",
        help="Path to save/load model file (.keras or .h5).",
    )
    parser.add_argument(
        "--output-dir",
        default="models/exported/tf_eval",
        help="Directory to save plots and reports.",
    )
    parser.add_argument("--image-path", default=None, help="Image path for --mode infer.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "train":
            train_pipeline(args)
        else:
            if not args.image_path:
                raise ValueError("--image-path is required when --mode infer")
            infer_pipeline(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
