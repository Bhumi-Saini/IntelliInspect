"""
IntelliInspect - YOLO Model Training & Fine-Tuning Pipeline
Trains or fine-tunes YOLO on industrial defect datasets (NEU-DET) with full metric logging.
"""

import os
import sys
import argparse
import shutil
import json

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODELS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ["YOLO_CONFIG_DIR"] = os.path.join(PROJECT_ROOT, ".yolo")
os.environ["YOLO_OFFLINE"] = "true"
os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

import torch
from ultralytics import YOLO
DEFAULT_DATA_YAML = os.path.join(PROJECT_ROOT, "dataset", "data.yaml")
WEIGHTS_DIR = os.path.join(MODELS_DIR, "weights")
METRICS_DIR = os.path.join(MODELS_DIR, "metrics")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def train_defect_detector(
    data_yaml: str = DEFAULT_DATA_YAML,
    base_model: str = "yolov8n.pt",
    epochs: int = 15,
    batch_size: int = 8,
    img_size: int = 640,
    device: str = "cpu",
    output_dir: str = os.path.join(MODELS_DIR, "runs")
) -> str:
    """
    Train or fine-tune YOLO detector on industrial surface defects.
    Saves best.pt to models/weights/best.pt and exports evaluation metrics.
    """
    print(f"=== [IntelliInspect] Initializing Training Pipeline ===")
    print(f"Data configuration: {data_yaml}")
    print(f"Base architecture: {base_model}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Image size: {img_size}")

    # Verify device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = "cpu"

    print(f"Compute Device: {device.upper()}")

    # Initialize YOLO model
    model = YOLO(base_model)

    # Execute training
    print("Beginning model training on industrial defect dataset...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name="defect_train",
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    # Locate and copy best weights
    run_dir = os.path.join(output_dir, "defect_train")
    best_weights_src = os.path.join(run_dir, "weights", "best.pt")
    target_weights = os.path.join(WEIGHTS_DIR, "best.pt")

    if os.path.exists(best_weights_src):
        shutil.copy(best_weights_src, target_weights)
        print(f"Saved best model weights to: {target_weights}")
    else:
        last_weights = os.path.join(run_dir, "weights", "last.pt")
        if os.path.exists(last_weights):
            shutil.copy(last_weights, target_weights)
            print(f"Copied last weights to: {target_weights}")

    # Copy plots & metrics to models/metrics
    for plot_file in ["confusion_matrix.png", "confusion_matrix_normalized.png", "results.png", "PR_curve.png", "results.csv"]:
        src = os.path.join(run_dir, plot_file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(METRICS_DIR, plot_file))

    # Evaluate validation metrics
    print("Evaluating trained model on validation split...")
    from models.evaluate import evaluate_model
    metrics = evaluate_model(weights_path=target_weights, data_yaml=data_yaml, device=device)

    print(f"=== [IntelliInspect] Training & Evaluation Complete! ===")
    print(f"mAP@50: {metrics.get('mAP50', 0):.4f} | Precision: {metrics.get('precision', 0):.4f} | Recall: {metrics.get('recall', 0):.4f}")
    return target_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IntelliInspect Industrial Defect Detector")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    train_defect_detector(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device
    )
