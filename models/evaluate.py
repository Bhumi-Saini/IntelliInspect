"""
IntelliInspect - Model Evaluation & Quantitative Metrics Suite
Evaluates trained YOLO weights against validation data and exports formal performance metrics.
"""

import os
import sys
import json
import argparse

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODELS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ["YOLO_CONFIG_DIR"] = os.path.join(PROJECT_ROOT, ".yolo")
os.environ["YOLO_OFFLINE"] = "true"
os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from typing import Dict, Any, Optional
import numpy as np
DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "weights", "best.pt")
DEFAULT_DATA_YAML = os.path.join(PROJECT_ROOT, "dataset", "data.yaml")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "metrics", "evaluation_metrics.json")
RUNS_DIR = os.path.join(MODELS_DIR, "runs", "val")

os.makedirs(os.path.dirname(METRICS_JSON_PATH), exist_ok=True)


def evaluate_model(
    weights_path: Optional[str] = None,
    data_yaml: Optional[str] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run quantitative validation of the trained YOLO model.
    Extracts Precision, Recall, F1, mAP50, mAP50-95, and per-class stats.
    """
    weights = weights_path or DEFAULT_WEIGHTS
    data = data_yaml or DEFAULT_DATA_YAML

    from ultralytics import YOLO

    if not os.path.exists(weights):
        # Fallback to yolov8n base model if custom best.pt does not exist
        print(f"Weights not found at {weights}. Falling back to base yolov8n.pt for evaluation...")
        weights = "yolov8n.pt"

    print(f"[IntelliInspect] Evaluating model weights: {weights} using data config: {data}")
    model = YOLO(weights)

    # Execute Ultralytics validation run
    val_results = model.val(
        data=data,
        device=device,
        project=os.path.join(MODELS_DIR, "runs"),
        name="defect_val",
        exist_ok=True,
        verbose=True
    )

    box = val_results.box

    # Global aggregate metrics
    map50 = float(box.map50) if hasattr(box, "map50") else 0.0
    map50_95 = float(box.map) if hasattr(box, "map") else 0.0
    precision = float(box.mp) if hasattr(box, "mp") else 0.0
    recall = float(box.mr) if hasattr(box, "mr") else 0.0

    # Calculate aggregate F1
    if precision + recall > 0:
        f1 = float(2 * (precision * recall) / (precision + recall))
    else:
        f1 = 0.0

    # Per-class breakdowns
    class_metrics = []
    names = val_results.names

    for i, class_name in names.items():
        try:
            p = float(box.p[i]) if hasattr(box, "p") and i < len(box.p) else precision
            r = float(box.r[i]) if hasattr(box, "r") and i < len(box.r) else recall
            f1_c = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            m50 = float(box.maps[i]) if hasattr(box, "maps") and i < len(box.maps) else map50
            m50_95 = float(box.maps[i]) if hasattr(box, "maps") and i < len(box.maps) else map50_95

            class_metrics.append({
                "Class": class_name.replace("_", " ").title(),
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "F1-Score": round(f1_c, 4),
                "mAP50": round(m50, 4),
                "mAP50-95": round(m50_95, 4)
            })
        except Exception as e:
            print(f"Error parsing metrics for class {class_name}: {e}")

    summary = {
        "model_weights": os.path.basename(weights),
        "data_config": os.path.basename(data),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "num_classes": len(class_metrics),
        "class_metrics": class_metrics
    }

    # Save to metrics JSON
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Metrics successfully saved to {METRICS_JSON_PATH}")
    return summary


def run_model_evaluation() -> Dict[str, Any]:
    """Helper wrapper callable by Streamlit components."""
    return evaluate_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IntelliInspect Model")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Path to model weights")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    evaluate_model(args.weights, args.data, args.device)
