"""
IntelliInspect - Inspection History Component
Audit logs, drill-down inspections, PDF generation, and CSV export.
"""

import os
import streamlit as st
import pandas as pd
from database.db import (
    get_all_inspections,
    get_inspection_details,
    delete_inspection,
    clear_all_records
)
from utils.pdf_report import generate_inspection_pdf
from utils.helpers import REPORTS_DIR


def render_history():
    """Render historical inspection logs with search, filter, and export capabilities."""
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; font-size: 26px; font-weight: 700; color: #f8fafc;">
                Inspection Audit History & Quality Logs
            </h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">
                Comprehensive compliance record of all processed manufacturing components.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Search and Filter Controls
    col_search, col_status, col_limit = st.columns([3, 2, 1])

    with col_search:
        search_query = st.text_input(
            "Search Records",
            placeholder="Search by ID, Image Name, or Defect Type...",
            label_visibility="collapsed"
        )

    with col_status:
        status_filter = st.selectbox(
            "Filter Status",
            options=["ALL", "PASS", "DEFECT DETECTED", "REVIEW REQUIRED"],
            index=0,
            label_visibility="collapsed"
        )

    with col_limit:
        records_limit = st.selectbox(
            "Limit",
            options=[20, 50, 100, 200],
            index=1,
            label_visibility="collapsed"
        )

    # Fetch records
    records = get_all_inspections(
        status_filter=status_filter if status_filter != "ALL" else None,
        search_query=search_query if search_query else None,
        limit=records_limit
    )

    if not records:
        st.info("No matching inspection records found in the database.")
        return

    # Export toolbar
    df_all = pd.DataFrame(records)
    csv_bytes = df_all.to_csv(index=False).encode("utf-8")

    col_info, col_export, col_clear = st.columns([4, 2, 2])
    with col_info:
        st.caption(f"Showing {len(records)} inspection record(s)")
    with col_export:
        st.download_button(
            label="Download History (CSV)",
            data=csv_bytes,
            file_name="intelliinspect_audit_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_clear:
        if st.button("Clear All History", type="secondary", use_container_width=True):
            if clear_all_records():
                st.success("History database cleared.")
                st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Render each inspection record as a card/expander
    for record in records:
        insp_id = record["inspection_id"]
        status = record["status"]
        timestamp = record["timestamp"]
        img_name = record["image_name"]
        defects = record["total_defects"]
        categories = record["defect_categories"]
        avg_conf = record["avg_confidence"]

        if status == "PASS":
            status_badge = '<span class="status-badge status-pass">PASS</span>'
        elif status == "DEFECT DETECTED":
            status_badge = '<span class="status-badge status-defect">DEFECT DETECTED</span>'
        else:
            status_badge = '<span class="status-badge status-review">REVIEW REQUIRED</span>'

        header_title = f"{insp_id} | {img_name} | {timestamp} | Defects: {defects}"

        with st.expander(header_title, expanded=False):
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <span style="font-weight: 600; font-size: 16px; color: #f8fafc;">{insp_id}</span>
                        <span style="margin-left: 10px;">{status_badge}</span>
                    </div>
                    <div style="color: #94a3b8; font-size: 13px;">
                        Timestamp: {timestamp}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            insp_data, defect_items = get_inspection_details(insp_id)

            # Details & Image Display
            img_col, meta_col = st.columns([1, 1])

            with img_col:
                res_path = record.get("result_image_path")
                orig_path = record.get("image_path")
                if res_path and os.path.exists(res_path):
                    st.image(res_path, caption=f"Defect Localization: {img_name}", use_container_width=True)
                elif orig_path and os.path.exists(orig_path):
                    st.image(orig_path, caption=f"Original: {img_name}", use_container_width=True)
                else:
                    st.caption("Image file no longer available on disk.")

            with meta_col:
                st.markdown(f"""
                    <div class="panel-box" style="margin-bottom: 12px;">
                        <div style="font-size: 13px; color: #94a3b8; margin-bottom: 4px;">IMAGE NAME</div>
                        <div style="font-size: 15px; font-weight: 600; color: #f1f5f9; word-break: break-all;">{img_name}</div>
                        <hr style="border-color: #334155; margin: 8px 0;">
                        <div style="font-size: 13px; color: #94a3b8; margin-bottom: 4px;">DEFECT CATEGORIES</div>
                        <div style="font-size: 14px; color: #f1f5f9;">{categories}</div>
                        <hr style="border-color: #334155; margin: 8px 0;">
                        <div style="font-size: 13px; color: #94a3b8; margin-bottom: 4px;">AVERAGE MODEL CONFIDENCE</div>
                        <div style="font-size: 15px; font-weight: 600; color: #38bdf8;">{avg_conf:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)

                # Generate and download PDF button
                pdf_filename = f"report_{insp_id}.pdf"
                pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

                # Generate on demand
                if st.button(f"Generate PDF Report ({insp_id})", key=f"gen_{insp_id}"):
                    try:
                        generate_inspection_pdf(insp_data, defect_items, pdf_path)
                        st.success("PDF Report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating PDF: {e}")

                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()
                    st.download_button(
                        label=f"Download PDF Audit ({insp_id})",
                        data=pdf_data,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        key=f"dl_{insp_id}",
                        use_container_width=True
                    )

            # Defect items breakdown table if defects were detected
            if defect_items:
                st.markdown("<h5 style='color: #f1f5f9; margin-top: 10px;'>Defect Coordinates & Classification</h5>", unsafe_allow_html=True)
                items_df = pd.DataFrame([
                    {
                        "Defect #": idx + 1,
                        "Class": item["defect_class"].upper(),
                        "Confidence": f"{item['confidence']:.1%}",
                        "X1 (px)": f"{item['bbox_x1']:.1f}",
                        "Y1 (px)": f"{item['bbox_y1']:.1f}",
                        "X2 (px)": f"{item['bbox_x2']:.1f}",
                        "Y2 (px)": f"{item['bbox_y2']:.1f}",
                        "Width (px)": f"{item['bbox_x2'] - item['bbox_x1']:.1f}",
                        "Height (px)": f"{item['bbox_y2'] - item['bbox_y1']:.1f}"
                    }
                    for idx, item in enumerate(defect_items)
                ])
                st.dataframe(items_df, use_container_width=True, hide_index=True)

            # Delete single button
            if st.button(f"Delete Record ({insp_id})", key=f"del_{insp_id}"):
                delete_inspection(insp_id)
                st.success(f"Record {insp_id} deleted.")
                st.rerun()
