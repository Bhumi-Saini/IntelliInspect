"""
IntelliInspect - Deep Learning Inference Engine
Performs object detection, bounding box localization, and classification using trained YOLO weights.
"""

import os
import sys

# Configure YOLO offline & writable config dir
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODELS_DIR)
os.environ["YOLO_CONFIG_DIR"] = os.path.join(PROJECT_ROOT, ".yolo")
os.environ["YOLO_OFFLINE"] = "true"
os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image
from preprocessing.image_processing import (
    load_image,
    assess_image_quality,
    apply_clahe,
    draw_detections,
    determine_inspection_status
)

DEFAULT_WEIGHTS_PATH = os.path.join(MODELS_DIR, "weights", "best.pt")

# Standard class mapping for industrial surface defects
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


class DefectDetector:
    """
    Industrial Object Detection Inference Pipeline using YOLO.
    """

    def __init__(self, weights_path: Optional[str] = None, conf_threshold: float = 0.35, iou_threshold: float = 0.45):
        self.weights_path = weights_path or DEFAULT_WEIGHTS_PATH
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model weights with graceful fallback."""
        try:
            from ultralytics import YOLO

            if os.path.exists(self.weights_path):
                print(f"[IntelliInspect] Loading trained model weights from: {self.weights_path}")
                self.model = YOLO(self.weights_path)
            else:
                # Use default yolov8n as base detector if custom weights have not been generated yet
                print(f"[IntelliInspect] Trained weights not found at {self.weights_path}. Initializing YOLO base model...")
                self.model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"[IntelliInspect] Error loading YOLO model: {e}")
            self.model = None

    def is_loaded(self) -> bool:
        """Check if model is successfully initialized."""
        return self.model is not None

    def predict(
        self,
        image_input: Union[str, bytes, np.ndarray, Image.Image],
        apply_enhancement: bool = False,
        conf_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run defect detection inference on a product image.

        Returns:
            dict containing:
                - 'annotated_image': np.ndarray (RGB with bounding boxes)
                - 'detections': List[Dict] (class, confidence, bbox, area)
                - 'total_defects': int
                - 'defect_categories': List[str]
                - 'avg_confidence': float
                - 'max_confidence': float
                - 'status': str ('PASS' | 'DEFECT DETECTED' | 'REVIEW REQUIRED')
                - 'status_reason': str
                - 'quality_assessment': Dict (blur, lighting)
        """
        # 1. Load image to standard RGB numpy array
        img_rgb = load_image(image_input)

        # 2. Assess image quality (blur and lighting)
        quality = assess_image_quality(img_rgb)

        # 3. Optional CLAHE contrast enhancement for hairline surface cracks
        process_img = apply_clahe(img_rgb) if apply_enhancement else img_rgb

        conf = conf_override if conf_override is not None else self.conf_threshold
        detections: List[Dict[str, Any]] = []

        if self.model is not None:
            try:
                # Run YOLO inference
                results = self.model.predict(
                    source=process_img,
                    conf=conf,
                    iou=self.iou_threshold,
                    verbose=False
                )

                if results and len(results) > 0:
                    r = results[0]
                    boxes = r.boxes

                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                        confidence = float(box.conf[0].cpu().numpy())
                        cls_idx = int(box.cls[0].cpu().numpy())

                        # Map class name
                        if hasattr(r, "names") and cls_idx in r.names:
                            class_name = r.names[cls_idx]
                        elif cls_idx < len(CLASSES):
                            class_name = CLASSES[cls_idx]
                        else:
                            class_name = f"defect_class_{cls_idx}"

                        x1, y1, x2, y2 = coords
                        area = max(0.0, (x2 - x1) * (y2 - y1))

                        detections.append({
                            "class": class_name,
                            "confidence": round(confidence, 4),
                            "bbox": [round(c, 2) for c in coords],
                            "area": round(area, 2)
                        })
            except Exception as e:
                print(f"[IntelliInspect] Inference execution error: {e}")

        # 4. Synthesize inspection outcome
        status, reason = determine_inspection_status(detections, quality, conf_threshold=conf)

        confidences = [d["confidence"] for d in detections]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        max_conf = max(confidences) if confidences else 0.0
        defect_categories = sorted(list(set(d["class"] for d in detections)))

        # 5. Draw bounding box visual overlays
        annotated = draw_detections(img_rgb, detections)

        return {
            "original_image": img_rgb,
            "annotated_image": annotated,
            "detections": detections,
            "total_defects": len(detections),
            "defect_categories": defect_categories,
            "avg_confidence": round(avg_conf, 4),
            "max_confidence": round(max_conf, 4),
            "status": status,
            "status_reason": reason,
            "quality_assessment": quality
        }


# Global singleton instance for application reuse
_detector_instance: Optional[DefectDetector] = None


def get_detector(weights_path: Optional[str] = None, conf_threshold: float = 0.35) -> DefectDetector:
    """Retrieve or initialize the singleton detector instance."""
    global _detector_instance
    if _detector_instance is None or (weights_path and _detector_instance.weights_path != weights_path):
        _detector_instance = DefectDetector(weights_path=weights_path, conf_threshold=conf_threshold)
    return _detector_instance
