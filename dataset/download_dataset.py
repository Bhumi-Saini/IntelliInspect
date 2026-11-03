"""
IntelliInspect - Dataset Preparation & Generator Module
Handles downloading, XML conversion, and generating calibrated industrial sample datasets.
"""

import os
import argparse
import random
import cv2
import numpy as np
from typing import Tuple, List, Dict


DEFECT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DATASET_ROOT)


def create_steel_texture(width: int = 640, height: int = 640) -> np.ndarray:
    """Generate realistic brushed steel/rolled metal background texture."""
    # Base metallic gray with slight gradient
    base_gray = random.randint(110, 160)
    img = np.full((height, width, 3), base_gray, dtype=np.uint8)

    # Rolling direction brushed striations (horizontal/vertical grain)
    grain = np.random.normal(0, random.uniform(4, 9), (height, width)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (1, 15), 0)

    # Add broad uneven illumination/shadowing
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    vignette = (1.0 - 0.15 * (xx**2 + yy**2)).astype(np.float32)

    img_float = (img.astype(np.float32) + grain[:, :, np.newaxis]) * vignette[:, :, np.newaxis]
    return np.clip(img_float, 20, 240).astype(np.uint8)


def inject_crazing(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
    """Inject branching network of fine micro-cracks (crazing)."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    num_cracks = random.randint(3, 7)
    for _ in range(num_cracks):
        pts = [(cx, cy)]
        curr_x, curr_y = cx, cy
        steps = random.randint(4, 8)
        for _ in range(steps):
            curr_x += random.randint(-18, 18)
            curr_y += random.randint(-18, 18)
            curr_x = np.clip(curr_x, x1, x2)
            curr_y = np.clip(curr_y, y1, y2)
            pts.append((int(curr_x), int(curr_y)))
        pts_arr = np.array(pts, np.int32)
        cv2.polylines(img, [pts_arr], isClosed=False, color=(35, 35, 35), thickness=1, lineType=cv2.LINE_AA)


def inject_inclusion(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
    """Inject dark non-metallic embedded impurities."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx = max(4, (x2 - x1) // 3)
    ry = max(4, (y2 - y1) // 3)
    # Dark core with diffused edge
    cv2.ellipse(img, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, (25, 25, 25), -1)
    cv2.ellipse(img, (cx, cy), (rx + 2, ry + 2), random.randint(0, 180), 0, 360, (50, 50, 50), 1)


def inject_patches(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
    """Inject irregular textural oxide blotch."""
    x1, y1, x2, y2 = bbox
    patch_w, patch_h = x2 - x1, y2 - y1
    mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    num_blobs = random.randint(4, 9)
    for _ in range(num_blobs):
        bx = random.randint(patch_w // 4, 3 * patch_w // 4)
        by = random.randint(patch_h // 4, 3 * patch_h // 4)
        br = random.randint(patch_w // 6, patch_w // 3)
        cv2.circle(mask, (bx, by), br, 255, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    sub = img[y1:y2, x1:x2].astype(np.float32)
    intensity_shift = random.choice([-35, 35])
    sub += (mask.astype(np.float32) / 255.0)[:, :, np.newaxis] * intensity_shift
    img[y1:y2, x1:x2] = np.clip(sub, 10, 245).astype(np.uint8)


def inject_pitted_surface(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
    """Inject cluster of micro-cavities/pits."""
    x1, y1, x2, y2 = bbox
    num_pits = random.randint(12, 30)
    for _ in range(num_pits):
        px = random.randint(x1, x2)
        py = random.randint(y1, y2)
        pr = random.randint(1, 4)
        cv2.circle(img, (px, py), pr, (30, 30, 30), -1)
        cv2.circle(img, (px + 1, py + 1), 1, (200, 200, 200), -1)  # highlight ring


def inject_rolled_in_scale(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
    """Inject elongated rolled-in oxide scale band."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = max(10, (x2 - x1) // 2)
    thickness = max(4, (y2 - y1) // 3)
    angle = random.randint(-15, 15)  # typically aligned along rolling axis
    cv2.ellipse(img, (cx, cy), (length, thickness), angle, 0, 360, (40, 40, 40), -1)


def inject_scratches(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
    """Inject sharp linear abrasion with shadow and highlight edges."""
    x1, y1, x2, y2 = bbox
    thickness = random.randint(1, 2)
    # Bright scoring line paired with dark shadow furrow
    cv2.line(img, (x1, y1), (x2, y2), (250, 250, 250), thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + 1, y1 + 1), (x2 + 1, y2 + 1), (30, 30, 30), thickness, cv2.LINE_AA)


INJECTORS = {
    "crazing": inject_crazing,
    "inclusion": inject_inclusion,
    "patches": inject_patches,
    "pitted_surface": inject_pitted_surface,
    "rolled-in_scale": inject_rolled_in_scale,
    "scratches": inject_scratches
}


def generate_defect_sample(
    width: int = 640,
    height: int = 640,
    min_defects: int = 1,
    max_defects: int = 3,
    specific_class: str = None
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """
    Generate synthetic industrial metal sheet image with calibrated defects and YOLO labels.
    Returns: (image_rgb, [(class_id, x_center, y_center, w, h)])
    """
    img = create_steel_texture(width, height)
    labels = []

    num_defects = random.randint(min_defects, max_defects)
    if num_defects == 0:
        return img, labels

    margin = 50
    for _ in range(num_defects):
        if specific_class and specific_class in DEFECT_CLASSES:
            cls_name = specific_class
        else:
            cls_name = random.choice(DEFECT_CLASSES)

        cls_id = DEFECT_CLASSES.index(cls_name)

        # Generate defect box within canvas
        box_w = random.randint(60, 160)
        box_h = random.randint(60, 160)
        box_x1 = random.randint(margin, width - margin - box_w)
        box_y1 = random.randint(margin, height - margin - box_h)
        box_x2 = box_x1 + box_w
        box_y2 = box_y1 + box_h

        # Inject defect artifact
        INJECTORS[cls_name](img, (box_x1, box_y1, box_x2, box_y2))

        # Convert to normalized YOLO coordinates
        x_center = (box_x1 + box_x2) / (2.0 * width)
        y_center = (box_y1 + box_y2) / (2.0 * height)
        norm_w = box_w / float(width)
        norm_h = box_h / float(height)

        labels.append((cls_id, x_center, y_center, norm_w, norm_h))

    return img, labels


def generate_sample_dataset(num_train: int = 60, num_val: int = 20) -> None:
    """Generate balanced training and validation splits with YOLO annotations."""
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_ROOT, "labels", split), exist_ok=True)

    samples_dir = os.path.join(PROJECT_ROOT, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    print(f"Generating {num_train} training samples and {num_val} validation samples...")

    # Training split
    for i in range(num_train):
        img, labels = generate_defect_sample(min_defects=1, max_defects=3)
        img_name = f"steel_train_{i:04d}.jpg"
        lbl_name = f"steel_train_{i:04d}.txt"

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DATASET_ROOT, "images", "train", img_name), img_bgr)

        with open(os.path.join(DATASET_ROOT, "labels", "train", lbl_name), "w") as f:
            for l in labels:
                f.write(f"{l[0]} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")

    # Validation split
    for i in range(num_val):
        img, labels = generate_defect_sample(min_defects=1, max_defects=3)
        img_name = f"steel_val_{i:04d}.jpg"
        lbl_name = f"steel_val_{i:04d}.txt"

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DATASET_ROOT, "images", "val", img_name), img_bgr)

        with open(os.path.join(DATASET_ROOT, "labels", "val", lbl_name), "w") as f:
            for l in labels:
                f.write(f"{l[0]} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")

    # Generate benchmark sample test images for the UI
    print("Generating dedicated benchmark samples for instant UI inspection...")
    for cls_name in DEFECT_CLASSES:
        img, _ = generate_defect_sample(min_defects=1, max_defects=2, specific_class=cls_name)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(samples_dir, f"sample_{cls_name}.jpg"), img_bgr)

    # Clean defect-free pass sample
    clean_img, _ = generate_defect_sample(min_defects=0, max_defects=0)
    cv2.imwrite(os.path.join(samples_dir, "sample_clean_pass.jpg"), cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))

    print(f"Dataset successfully prepared in {DATASET_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntelliInspect Dataset Tool")
    parser.add_argument("--train-count", type=int, default=60, help="Number of training images")
    parser.add_argument("--val-count", type=int, default=20, help="Number of validation images")
    args = parser.parse_args()

    generate_sample_dataset(args.train_count, args.val_count)
