"""
IntelliInspect - Model Performance & Validation Analytics Component
Visualizes genuine benchmark metrics, mAP50, mAP50-95, precision-recall curve, and confusion matrix.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import PROJECT_ROOT


METRICS_FILE = os.path.join(PROJECT_ROOT, "models", "metrics", "evaluation_metrics.json")
CONFUSION_MATRIX_IMAGE = os.path.join(PROJECT_ROOT, "models", "metrics", "confusion_matrix.png")
PR_CURVE_IMAGE = os.path.join(PROJECT_ROOT, "models", "metrics", "PR_curve.png")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "models", "metrics", "results.csv")


def load_metrics_data():
    """Load evaluation metrics JSON if available."""
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading evaluation metrics: {e}")
            return None
    return None


def render_performance():
    """Render the Model Performance & Verification page."""
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; font-size: 26px; font-weight: 700; color: #f8fafc;">
                Computer Vision Model Evaluation & Benchmark Metrics
            </h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">
                Quantitative verification of the fine-tuned YOLO object detection architecture on industrial test samples.
            </p>
        </div>
    """, unsafe_allow_html=True)

    metrics_data = load_metrics_data()

    if not metrics_data:
        st.warning("""
            **No evaluation results found on disk.**
            
            To populate authentic model evaluation metrics:
            1. Ensure the dataset is prepared in `dataset/`
            2. Run evaluation via `python models/evaluate.py` or click the button below.
        """)

        col_eval, col_dummy = st.columns([1, 2])
        with col_eval:
            if st.button("Run Evaluation Pipeline Now", type="primary"):
                with st.spinner("Running model evaluation on validation split..."):
                    try:
                        from models.evaluate import run_model_evaluation
                        res = run_model_evaluation()
                        st.success("Evaluation complete! Reloading metrics...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Evaluation error: {e}")
        return

    # Top summary metrics cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">mAP @ 0.50</div>
                <div class="metric-value" style="color: #38bdf8;">{metrics_data.get('mAP50', 0.0):.1%}</div>
                <div class="metric-subtext">Mean Avg Precision</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">mAP @ 0.50:0.95</div>
                <div class="metric-value" style="color: #818cf8;">{metrics_data.get('mAP50_95', 0.0):.1%}</div>
                <div class="metric-subtext">Strict IoU threshold</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PRECISION</div>
                <div class="metric-value" style="color: #22c55e;">{metrics_data.get('precision', 0.0):.1%}</div>
                <div class="metric-subtext">True positive ratio</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RECALL</div>
                <div class="metric-value" style="color: #f59e0b;">{metrics_data.get('recall', 0.0):.1%}</div>
                <div class="metric-subtext">Defect sensitivity</div>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1-SCORE</div>
                <div class="metric-value" style="color: #ec4899;">{metrics_data.get('f1_score', 0.0):.1%}</div>
                <div class="metric-subtext">Harmonic mean</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Class-wise Metrics Table
    st.markdown("<h4 style='color: #f1f5f9; margin-bottom: 8px;'>Class-wise Detection Performance</h4>", unsafe_allow_html=True)
    class_metrics = metrics_data.get("class_metrics", [])
    if class_metrics:
        df_classes = pd.DataFrame(class_metrics)
        # Format percentages
        format_cols = ["Precision", "Recall", "F1-Score", "mAP50", "mAP50-95"]
        for c in format_cols:
            if c in df_classes.columns:
                df_classes[c] = df_classes[c].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
        st.dataframe(df_classes, use_container_width=True, hide_index=True)

    # Evaluation Charts / Visualizations
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("<h4 style='color: #f1f5f9; margin-bottom: 8px;'>Per-Class mAP@0.5 Comparison</h4>", unsafe_allow_html=True)
        if class_metrics:
            raw_classes = pd.DataFrame(metrics_data.get("class_metrics", []))
            fig = px.bar(
                raw_classes,
                x="Class",
                y="mAP50",
                color="mAP50",
                color_continuous_scale="Viridis",
                range_y=[0, 1.05],
                text_auto=".1%"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                yaxis=dict(gridcolor="#334155", title="mAP @ 0.50"),
                xaxis=dict(title="Defect Category"),
                coloraxis_showscale=False,
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        st.markdown("<h4 style='color: #f1f5f9; margin-bottom: 8px;'>Precision vs Recall Trade-off</h4>", unsafe_allow_html=True)
        if class_metrics:
            raw_classes = pd.DataFrame(metrics_data.get("class_metrics", []))
            fig_pr = px.scatter(
                raw_classes,
                x="Recall",
                y="Precision",
                text="Class",
                color="F1-Score",
                size=[20] * len(raw_classes),
                color_continuous_scale="Turbo",
                range_x=[0, 1.05],
                range_y=[0, 1.05]
            )
            fig_pr.update_traces(textposition="top center")
            fig_pr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="#334155", title="Recall"),
                yaxis=dict(gridcolor="#334155", title="Precision"),
                coloraxis_showscale=False,
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_pr, use_container_width=True, config={"displayModeBar": False})

    # Confusion Matrix & Curves if available
    if os.path.exists(CONFUSION_MATRIX_IMAGE) or os.path.exists(PR_CURVE_IMAGE):
        st.markdown("<h4 style='color: #f1f5f9; margin-top: 15px;'>Diagnostic Plots & Error Analysis</h4>", unsafe_allow_html=True)
        img_c1, img_c2 = st.columns(2)
        with img_c1:
            if os.path.exists(CONFUSION_MATRIX_IMAGE):
                st.image(CONFUSION_MATRIX_IMAGE, caption="Validation Confusion Matrix", use_container_width=True)
        with img_c2:
            if os.path.exists(PR_CURVE_IMAGE):
                st.image(PR_CURVE_IMAGE, caption="Precision-Recall Curve (IoU=0.5)", use_container_width=True)

    # Re-evaluate button
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    if st.button("Re-evaluate Model on Test Split"):
        with st.spinner("Re-calculating validation metrics..."):
            try:
                from models.evaluate import run_model_evaluation
                run_model_evaluation()
                st.success("Evaluation refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Error during re-evaluation: {e}")
