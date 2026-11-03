"""
IntelliInspect - Image Inspection & Defect Analysis Component
Single-image, batch processing, and live webcam inspection with visual defect localization.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from models.predict import get_detector
from preprocessing.image_processing import (
    load_image,
    assess_image_quality,
    apply_clahe,
    DEFECT_COLORS
)
from database.db import save_inspection
from utils.helpers import (
    generate_inspection_id,
    save_uploaded_file,
    save_result_image,
    REPORTS_DIR,
    SAMPLES_DIR
)
from utils.pdf_report import generate_inspection_pdf


def render_inspection():
    """Render interactive product inspection interface."""
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; font-size: 26px; font-weight: 700; color: #f8fafc;">
                Automated Industrial Defect Inspection
            </h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">
                Execute deep learning defect localization, quality verification, and automated compliance auditing.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Inspection Mode Selection
    mode = st.radio(
        "Inspection Mode",
        ["Single Product Inspection", "Batch Image Inspection", "Real-Time / Camera Inspection"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if mode == "Single Product Inspection":
        render_single_inspection()
    elif mode == "Batch Image Inspection":
        render_batch_inspection()
    else:
        render_webcam_inspection()


def render_single_inspection():
    """Single product image inspection pipeline."""
    # Control Sidebar / Settings Expander
    with st.expander("Inspection Parameters & Image Preprocessing", expanded=False):
        c_conf, c_clahe, c_iou = st.columns(3)
        with c_conf:
            conf_thresh = st.slider(
                "Detection Confidence Threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.35,
                step=0.05,
                help="Minimum model confidence required to register a defect bounding box."
            )
        with c_clahe:
            enable_clahe = st.checkbox(
                "Apply CLAHE Contrast Boost",
                value=False,
                help="Enhances micro-fractures, subtle hairline cracks, and low-contrast surface abrasions."
            )
        with c_iou:
            iou_thresh = st.slider(
                "NMS IoU Threshold",
                min_value=0.20,
                max_value=0.80,
                value=0.45,
                step=0.05,
                help="Non-Maximum Suppression threshold for overlapping bounding boxes."
            )

    # Input Selection: File Upload OR Select Benchmark Sample
    col_input_choice, col_samples = st.columns([1, 1])

    with col_input_choice:
        uploaded_file = st.file_uploader(
            "Upload Manufactured Product Surface Image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG. Min recommended resolution: 200x200."
        )

    # Sample library selector
    sample_files = []
    if os.path.exists(SAMPLES_DIR):
        sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".png"))]

    selected_sample = None
    with col_samples:
        st.markdown("<p style='font-size: 14px; color: #94a3b8; margin-bottom: 8px;'>Or choose from verified factory test benchmarks:</p>", unsafe_allow_html=True)
        if sample_files:
            sample_options = ["None (Use uploaded file)"] + sample_files
            choice = st.selectbox("Select Test Benchmark", options=sample_options, index=0)
            if choice != "None (Use uploaded file)":
                selected_sample = os.path.join(SAMPLES_DIR, choice)
        else:
            st.caption("No sample benchmarks generated yet. Upload an image above.")

    # Determine active image source
    active_image = None
    image_name = "sample_inspection.jpg"

    if uploaded_file is not None:
        active_image = Image.open(uploaded_file)
        image_name = uploaded_file.name
    elif selected_sample is not None:
        active_image = Image.open(selected_sample)
        image_name = os.path.basename(selected_sample)

    if active_image is None:
        st.info("Please upload an image or choose a factory benchmark sample from the dropdown to proceed.")
        return

    # Image Quality Pre-check
    img_np = np.array(active_image.convert("RGB"))
    quality_check = assess_image_quality(img_np)

    # Layout: Original Image & Quality Diagnostics
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    col_prev, col_diag = st.columns([1, 1])

    with col_prev:
        st.markdown("<h4 style='color: #f1f5f9; font-size: 16px; margin-bottom: 8px;'>Product Surface Preview</h4>", unsafe_allow_html=True)
        display_img = apply_clahe(img_np) if enable_clahe else img_np
        st.image(display_img, caption=f"Active Sample: {image_name} ({img_np.shape[1]}x{img_np.shape[0]} px)", use_container_width=True)

    with col_diag:
        st.markdown("<h4 style='color: #f1f5f9; font-size: 16px; margin-bottom: 8px;'>Optical Quality Pre-check</h4>", unsafe_allow_html=True)

        is_acc = quality_check["is_acceptable"]
        badge_style = "background: #064e3b; color: #34d399;" if is_acc else "background: #78350f; color: #fbbf24;"
        badge_text = "PASSED QUALITY AUDIT" if is_acc else "OPTICAL WARNING"

        st.markdown(f"""
            <div class="panel-box">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-weight: 600; color: #f1f5f9;">Optical Calibration Status</span>
                    <span style="font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; {badge_style}">{badge_text}</span>
                </div>
                <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 8px;">
                    <b>Condition:</b> {quality_check['condition']}
                </div>
                <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 8px;">
                    <b>Focus Sharpness (Laplacian Var):</b> {quality_check['blur_score']} <i>(Threshold: 80.0)</i>
                </div>
                <div style="font-size: 13px; color: #cbd5e1;">
                    <b>Mean Surface Luminescence:</b> {quality_check['brightness']} / 255.0
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Trigger Button
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        inspect_button = st.button("RUN INTELLIINSPECT AI AUDIT", type="primary", use_container_width=True)

    if inspect_button or st.session_state.get(f"inspected_{image_name}", False):
        st.session_state[f"inspected_{image_name}"] = True

        with st.spinner("Executing YOLO defect detection & classification..."):
            detector = get_detector(conf_threshold=conf_thresh)
            results = detector.predict(
                image_input=img_np,
                apply_enhancement=enable_clahe,
                conf_override=conf_thresh
            )

        st.markdown("<hr style='border-color: #334155; margin: 24px 0;'>", unsafe_allow_html=True)

        # Render Results Section
        status = results["status"]
        status_reason = results["status_reason"]
        total_defects = results["total_defects"]
        avg_conf = results["avg_confidence"]
        max_conf = results["max_confidence"]
        categories = results["defect_categories"]
        detections = results["detections"]

        # Status Banner
        if status == "PASS":
            banner_class = "status-pass"
            banner_icon = "COMPLIANT PASS"
            banner_desc = "Product surface satisfies all visual quality tolerance specifications. No structural or textural defects detected."
        elif status == "DEFECT DETECTED":
            banner_class = "status-defect"
            banner_icon = "DEFECT DETECTED - REJECT / REWORK"
            banner_desc = f"Identified {total_defects} defect instance(s) requiring line intervention: {', '.join(categories)}."
        else:
            banner_class = "status-review"
            banner_icon = "REVIEW REQUIRED"
            banner_desc = status_reason

        st.markdown(f"""
            <div class="status-banner {banner_class}">
                <div style="font-size: 20px; font-weight: 700; margin-bottom: 4px;">{banner_icon}</div>
                <div style="font-size: 14px; opacity: 0.95;">{banner_desc}</div>
            </div>
        """, unsafe_allow_html=True)

        # 4 Inspection Metrics Cards
        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        with res_c1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">DEFECT COUNT</div>
                    <div class="metric-value">{total_defects}</div>
                    <div class="metric-subtext">Discrete bounding regions</div>
                </div>
            """, unsafe_allow_html=True)
        with res_c2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">DEFECT CLASSES</div>
                    <div class="metric-value" style="font-size: 18px;">{len(categories)}</div>
                    <div class="metric-subtext">{', '.join(categories) if categories else 'None'}</div>
                </div>
            """, unsafe_allow_html=True)
        with res_c3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">AVERAGE CONFIDENCE</div>
                    <div class="metric-value" style="color: #38bdf8;">{avg_conf:.1%}</div>
                    <div class="metric-subtext">Model certainty</div>
                </div>
            """, unsafe_allow_html=True)
        with res_c4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MAX CONFIDENCE</div>
                    <div class="metric-value" style="color: #818cf8;">{max_conf:.1%}</div>
                    <div class="metric-subtext">Peak detection score</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Visual Comparison (Before and After)
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("<h4 style='color: #f1f5f9; font-size: 16px;'>Original Manufactured Surface</h4>", unsafe_allow_html=True)
            st.image(results["original_image"], use_container_width=True)

        with img_col2:
            st.markdown("<h4 style='color: #f1f5f9; font-size: 16px;'>Defect Localization (Bounding Boxes)</h4>", unsafe_allow_html=True)
            st.image(results["annotated_image"], use_container_width=True)

        # Defect Table
        if detections:
            st.markdown("<h4 style='color: #f1f5f9; margin-top: 15px;'>Detected Defect Instances & Coordinates</h4>", unsafe_allow_html=True)
            det_df = pd.DataFrame([
                {
                    "Item #": idx + 1,
                    "Defect Category": d["class"].replace("_", " ").title(),
                    "Model Confidence": f"{d['confidence']:.1%}",
                    "Bounding Box [X1, Y1, X2, Y2]": f"[{d['bbox'][0]:.1f}, {d['bbox'][1]:.1f}, {d['bbox'][2]:.1f}, {d['bbox'][3]:.1f}]",
                    "Defect Area (px²)": f"{d['area']:.1f}",
                    "Estimated Severity": "Critical" if d["class"] in ["inclusion", "crazing"] else ("High" if d["class"] in ["scratches", "rolled-in_scale"] else "Medium")
                }
                for idx, d in enumerate(detections)
            ])
            st.dataframe(det_df, use_container_width=True, hide_index=True)

        # Persistence to Database & Report Generation
        insp_id = generate_inspection_id()

        # Save images to disk
        orig_save_path = save_result_image(results["original_image"], prefix="orig")
        annotated_save_path = save_result_image(results["annotated_image"], prefix="annotated")

        # Save to SQLite
        save_db_btn = st.button("Save Inspection to Audit History Database", type="secondary")
        if save_db_btn or st.session_state.get(f"saved_{image_name}", False):
            st.session_state[f"saved_{image_name}"] = True
            saved = save_inspection(
                inspection_id=insp_id,
                image_name=image_name,
                image_path=orig_save_path,
                result_image_path=annotated_save_path,
                total_defects=total_defects,
                defect_categories=categories,
                avg_confidence=avg_conf,
                max_confidence=max_conf,
                status=status,
                defect_items=detections,
                notes=status_reason
            )
            if saved:
                st.success(f"Inspection saved to SQLite database with ID: **{insp_id}**")

        # Generate PDF Report
        pdf_path = os.path.join(REPORTS_DIR, f"audit_{insp_id}.pdf")
        insp_dict = {
            "inspection_id": insp_id,
            "image_name": image_name,
            "status": status,
            "total_defects": total_defects,
            "defect_categories": ", ".join(categories) if categories else "None",
            "avg_confidence": avg_conf,
            "max_confidence": max_conf,
            "result_image_path": annotated_save_path,
            "notes": status_reason
        }

        try:
            generate_inspection_pdf(insp_dict, detections, pdf_path)
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="Download Formal PDF Audit Report",
                        data=pdf_file.read(),
                        file_name=f"IntelliInspect_Report_{insp_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        except Exception as e:
            st.caption(f"PDF generation note: {e}")


def render_batch_inspection():
    """Batch inspection for processing multiple products simultaneously."""
    st.markdown("<h4 style='color: #f1f5f9;'>Batch Industrial Inspection</h4>", unsafe_allow_html=True)
    st.caption("Upload multiple product surface images to execute high-throughput quality analysis.")

    files = st.file_uploader(
        "Upload Image Batch (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if not files:
        st.info("Upload 2 or more images to run batch inspection.")
        return

    st.markdown(f"**Loaded {len(files)} image(s) for batch inspection.**")
    batch_btn = st.button("Start Batch Quality Audit", type="primary")

    if batch_btn:
        detector = get_detector()
        results_list = []
        progress_bar = st.progress(0)

        for i, f in enumerate(files):
            img_pil = Image.open(f)
            img_np = np.array(img_pil.convert("RGB"))
            res = detector.predict(img_np)
            insp_id = generate_inspection_id()

            orig_path = save_result_image(res["original_image"], prefix="batch_orig")
            annot_path = save_result_image(res["annotated_image"], prefix="batch_res")

            save_inspection(
                inspection_id=insp_id,
                image_name=f.name,
                image_path=orig_path,
                result_image_path=annot_path,
                total_defects=res["total_defects"],
                defect_categories=res["defect_categories"],
                avg_confidence=res["avg_confidence"],
                max_confidence=res["max_confidence"],
                status=res["status"],
                defect_items=res["detections"],
                notes=res["status_reason"]
            )

            results_list.append({
                "Inspection ID": insp_id,
                "File Name": f.name,
                "Status": res["status"],
                "Total Defects": res["total_defects"],
                "Categories": ", ".join(res["defect_categories"]) if res["defect_categories"] else "None",
                "Avg Confidence": f"{res['avg_confidence']:.1%}"
            })

            progress_bar.progress((i + 1) / len(files))

        st.success(f"Batch audit completed! Processed {len(files)} components.")
        df_batch = pd.DataFrame(results_list)
        st.dataframe(df_batch, use_container_width=True, hide_index=True)


def render_webcam_inspection():
    """Real-time optical inspection via camera input."""
    st.markdown("<h4 style='color: #f1f5f9;'>Real-Time Live Camera Optical Inspection</h4>", unsafe_allow_html=True)
    st.caption("Capture product surface frames from an integrated inspection webcam or optical sensor.")

    camera_image = st.camera_input("Capture Product Frame for Optical Inspection")

    if camera_image is not None:
        img_pil = Image.open(camera_image)
        img_np = np.array(img_pil.convert("RGB"))

        with st.spinner("Analyzing camera frame..."):
            detector = get_detector()
            res = detector.predict(img_np)

        c1, c2 = st.columns(2)
        with c1:
            st.image(res["original_image"], caption="Captured Frame", use_container_width=True)
        with c2:
            st.image(res["annotated_image"], caption=f"Status: {res['status']}", use_container_width=True)

        st.markdown(f"**Verdict:** {res['status']} | **Defects:** {res['total_defects']} | **Categories:** {', '.join(res['defect_categories']) if res['defect_categories'] else 'None'}")
