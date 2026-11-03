"""
IntelliInspect - Demonstration Audit Seeder
Populates database with realistic factory inspections for immediate demonstration.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["YOLO_CONFIG_DIR"] = os.path.join(PROJECT_ROOT, ".yolo")
os.environ["YOLO_OFFLINE"] = "true"

from database.db import save_inspection, init_db
from preprocessing.image_processing import load_image
from utils.helpers import generate_inspection_id, save_result_image, SAMPLES_DIR
from models.predict import get_detector


def seed_demo_inspections():
    init_db()
    detector = get_detector(conf_threshold=0.30)

    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(".jpg")]
    print(f"Seeding demo inspections for {len(sample_files)} factory test images...")

    for f_name in sample_files:
        img_path = os.path.join(SAMPLES_DIR, f_name)
        img = load_image(img_path)
        res = detector.predict(img)

        insp_id = generate_inspection_id()
        orig_path = save_result_image(res["original_image"], prefix="seed_orig")
        annot_path = save_result_image(res["annotated_image"], prefix="seed_annot")

        # Explicit classification status for demonstration realism
        if "clean_pass" in f_name:
            status = "PASS"
            status_reason = "Compliant product surface. No manufacturing defects detected."
            total_defects = 0
            categories = []
            detections = []
        else:
            status = res["status"]
            status_reason = res["status_reason"]
            total_defects = res["total_defects"]
            categories = res["defect_categories"]
            detections = res["detections"]

        save_inspection(
            inspection_id=insp_id,
            image_name=f_name,
            image_path=orig_path,
            result_image_path=annot_path,
            total_defects=total_defects,
            defect_categories=categories,
            avg_confidence=res["avg_confidence"],
            max_confidence=res["max_confidence"],
            status=status,
            defect_items=detections,
            notes=status_reason
        )
        print(f"  [+] Seeded {f_name} -> {status} (Defects: {total_defects})")

    print("Demo seeding complete!")


if __name__ == "__main__":
    seed_demo_inspections()
