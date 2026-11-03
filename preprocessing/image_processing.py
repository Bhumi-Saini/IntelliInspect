"""
IntelliInspect - Image Preprocessing & Computer Vision Pipeline
Industrial-grade image normalization, defect enhancement (CLAHE), blur/lighting checks,
and bounding box visualization utilities.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from typing import Tuple, Dict, Any, List, Optional


# Palette for industrial defect classes
# Professional color coding for distinct defect categories
DEFECT_COLORS = {
    "crazing": (230, 81, 0),         # Amber/Orange
    "inclusion": (194, 24, 91),       # Deep Pink/Magenta
    "patches": (56, 142, 60),         # Green
    "pitted_surface": (123, 31, 162), # Deep Purple
    "rolled-in_scale": (2, 136, 209), # Cerulean Blue
    "scratches": (211, 47, 47),       # Strong Red
    "crack": (229, 57, 53),           # Bright Red
    "dent": (245, 124, 0),            # Dark Orange
    "surface_damage": (109, 76, 65),  # Brown
    "unknown": (97, 97, 97)           # Gray
}


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """Return RGB color tuple for a defect class."""
    norm = class_name.lower().replace(" ", "_").replace("-", "_")
    for key, color in DEFECT_COLORS.items():
        if key in norm or norm in key:
            return color
    return DEFECT_COLORS["unknown"]


def load_image(image_input) -> np.ndarray:
    """
    Safely load an image from filepath, bytes, PIL Image, or numpy array.
    Returns RGB numpy array (H, W, C).
    """
    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        elif image_input.shape[2] == 3:
            return image_input
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))
    elif isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str) and os.path.exists(image_input):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Failed to load image from path: {image_input}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from bytes")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel of LAB color space.
    Significantly enhances faint scratches, micro-cracks, and surface texture defects.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def assess_image_quality(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Perform quality assessment on the uploaded product image:
    - Blur score (Laplacian variance)
    - Mean brightness (0-255)
    - Exposure condition (NORMAL, UNDEREXPOSED, OVEREXPOSED, BLURRED)
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    is_blurred = blur_score < 40.0
    is_underexposed = brightness < 45.0
    is_overexposed = brightness > 225.0

    if is_blurred:
        condition = "BLURRED / LOW RESOLUTION"
    elif is_underexposed:
        condition = "UNDEREXPOSED / LOW LIGHT"
    elif is_overexposed:
        condition = "OVEREXPOSED / SEVERE GLARE"
    else:
        condition = "OPTIMAL"

    return {
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "condition": condition,
        "is_acceptable": not (is_blurred or is_underexposed or is_overexposed)
    }


def letterbox_resize(img: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with aspect ratio preservation and letterbox padding for neural network input.
    Returns: padded_image, scale_factor, (pad_w, pad_h)
    """
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return canvas, scale, (pad_w, pad_h)


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    line_thickness: int = 3,
    show_labels: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes and class badges onto the product image with professional styling.
    detections is a list of dicts: {'class': str, 'confidence': float, 'bbox': [x1, y1, x2, y2]}
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Dynamically scale font & line thickness to image dimensions
    scale_factor = max(w, h) / 1000.0
    box_thick = max(2, int(line_thickness * scale_factor))
    font_scale = max(0.45, 0.6 * scale_factor)

    for det in detections:
        cls_name = det.get("class", "Defect")
        conf = float(det.get("confidence", 0.0))
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]

        # Clamp coordinates to image dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color_rgb = get_class_color(cls_name)
        # Note: OpenCV drawing functions use BGR if image is BGR, or RGB if image is RGB
        # We work in RGB throughout the pipeline
        color = color_rgb

        # Draw outer rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thick)

        # Draw corner accents for high-tech aesthetic
        corner_len = min(20, max(8, int((x2 - x1) * 0.15)))
        accent_thick = box_thick + 1
        # Top-left corner
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, accent_thick)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, accent_thick)
        # Top-right corner
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, accent_thick)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, accent_thick)
        # Bottom-left corner
        cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, accent_thick)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, accent_thick)
        # Bottom-right corner
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, accent_thick)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, accent_thick)

        if show_labels:
            label = f"{cls_name.upper()} {conf:.1%}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

            # Label box above bounding box, or below if near top
            label_y1 = max(0, y1 - text_h - 10)
            label_y2 = y1 if y1 - text_h - 10 >= 0 else y1 + text_h + 10
            label_x2 = min(w - 1, x1 + text_w + 10)

            # Draw background fill for text
            cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
            # Draw text in white or dark depending on color brightness
            text_pos_y = label_y2 - 4 if y1 - text_h - 10 >= 0 else label_y2 - 4
            cv2.putText(
                annotated,
                label,
                (x1 + 4, text_pos_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return annotated


def determine_inspection_status(detections: List[Dict[str, Any]], quality_check: Dict[str, Any], conf_threshold: float = 0.35) -> Tuple[str, str]:
    """
    Determine inspection outcome:
    - PASS: No defects detected with acceptable image quality.
    - DEFECT DETECTED: 1 or more defects detected with high confidence.
    - REVIEW REQUIRED: Model confidence is uncertain, or image quality is compromised.
    Returns: (status, reasoning)
    """
    if not quality_check.get("is_acceptable", True):
        return "REVIEW REQUIRED", f"Image quality warning: {quality_check.get('condition', 'Suboptimal lighting or blur')}."

    if not detections:
        return "PASS", "No manufacturing defects detected. Product meets visual standards."

    # Check confidences
    confidences = [d.get("confidence", 0.0) for d in detections]
    min_conf = min(confidences) if confidences else 0.0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # If detection confidence is borderline, review is advisable
    if avg_conf < 0.40:
        return "REVIEW REQUIRED", f"Detected {len(detections)} defect(s), but average confidence is borderline ({avg_conf:.1%}). Human review recommended."

    return "DEFECT DETECTED", f"Identified {len(detections)} defect(s) with high confidence ({avg_conf:.1%} avg)."
