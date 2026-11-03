"""
IntelliInspect - Comprehensive System Integration Test
Verifies end-to-end functionality: database, image processing, model inference, PDF generation.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["YOLO_CONFIG_DIR"] = os.path.join(PROJECT_ROOT, ".yolo")
os.environ["YOLO_OFFLINE"] = "true"

from database.db import init_db, save_inspection, get_dashboard_metrics, get_all_inspections
from preprocessing.image_processing import load_image, assess_image_quality, apply_clahe, draw_detections
from utils.helpers import generate_inspection_id, save_result_image, SAMPLES_DIR, REPORTS_DIR
from utils.pdf_report import generate_inspection_pdf
from models.predict import get_detector


def run_system_tests():
    print("=" * 60)
    print("INTELLIINSPECT END-TO-END SYSTEM INTEGRATION TEST")
    print("=" * 60)

    # 1. Test Database
    print("\n[1/5] Testing SQLite Database...")
    init_db()
    metrics_before = get_dashboard_metrics()
    print(f"Initial DB state: {metrics_before}")

    # 2. Test Image Processing
    print("\n[2/5] Testing Image Preprocessing & Optical Quality Guard...")
    sample_img_path = os.path.join(SAMPLES_DIR, "sample_scratches.jpg")
    if not os.path.exists(sample_img_path):
        from dataset.download_dataset import generate_sample_dataset
        generate_sample_dataset(num_train=10, num_val=5)

    img = load_image(sample_img_path)
    quality = assess_image_quality(img)
    enhanced = apply_clahe(img)
    print(f"Loaded image shape: {img.shape}")
    print(f"Quality assessment: {quality}")
    print(f"CLAHE enhanced shape: {enhanced.shape}")
    assert img.shape == enhanced.shape, "Shape mismatch after CLAHE"

    # 3. Test Model Inference
    print("\n[3/5] Testing YOLO Model Inference...")
    detector = get_detector(conf_threshold=0.25)
    results = detector.predict(img)
    print(f"Inference Verdict: {results['status']}")
    print(f"Total Defects Found: {results['total_defects']}")
    print(f"Categories: {results['defect_categories']}")
    print(f"Avg Confidence: {results['avg_confidence']:.2%}")

    # 4. Test Result Persistence
    print("\n[4/5] Testing Database Persistence...")
    insp_id = generate_inspection_id()
    orig_path = save_result_image(results["original_image"], prefix="test_orig")
    res_path = save_result_image(results["annotated_image"], prefix="test_res")

    saved = save_inspection(
        inspection_id=insp_id,
        image_name="sample_scratches.jpg",
        image_path=orig_path,
        result_image_path=res_path,
        total_defects=results["total_defects"],
        defect_categories=results["defect_categories"],
        avg_confidence=results["avg_confidence"],
        max_confidence=results["max_confidence"],
        status=results["status"],
        defect_items=results["detections"],
        notes=results["status_reason"]
    )
    assert saved, "Failed to persist inspection record to SQLite"
    print(f"Successfully saved inspection record: {insp_id}")

    # Seed clean pass sample as well
    pass_img_path = os.path.join(SAMPLES_DIR, "sample_clean_pass.jpg")
    if os.path.exists(pass_img_path):
        pass_img = load_image(pass_img_path)
        pass_res = detector.predict(pass_img)
        pass_insp_id = generate_inspection_id()
        pass_orig = save_result_image(pass_res["original_image"], prefix="pass_orig")
        pass_annot = save_result_image(pass_res["annotated_image"], prefix="pass_res")
        save_inspection(
            inspection_id=pass_insp_id,
            image_name="sample_clean_pass.jpg",
            image_path=pass_orig,
            result_image_path=pass_annot,
            total_defects=pass_res["total_defects"],
            defect_categories=pass_res["defect_categories"],
            avg_confidence=pass_res["avg_confidence"],
            max_confidence=pass_res["max_confidence"],
            status=pass_res["status"],
            defect_items=pass_res["detections"],
            notes=pass_res["status_reason"]
        )
        print(f"Successfully seeded clean pass record: {pass_insp_id}")

    metrics_after = get_dashboard_metrics()
    print(f"Updated DB Metrics: {metrics_after}")

    # 5. Test PDF Generation
    print("\n[5/5] Testing PDF Report Engine...")
    pdf_path = os.path.join(REPORTS_DIR, f"test_report_{insp_id}.pdf")
    insp_dict = {
        "inspection_id": insp_id,
        "image_name": "sample_scratches.jpg",
        "status": results["status"],
        "total_defects": results["total_defects"],
        "defect_categories": ", ".join(results["defect_categories"]),
        "avg_confidence": results["avg_confidence"],
        "max_confidence": results["max_confidence"],
        "result_image_path": res_path,
        "notes": results["status_reason"]
    }
    generated_pdf = generate_inspection_pdf(insp_dict, results["detections"], pdf_path)
    assert os.path.exists(generated_pdf), "PDF file was not created"
    print(f"Generated Audit PDF: {generated_pdf} (Size: {os.path.getsize(generated_pdf):,} bytes)")

    print("\n" + "=" * 60)
    print("ALL 5 SUBSYSTEM INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_system_tests()
