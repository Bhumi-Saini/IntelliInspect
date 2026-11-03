"""
IntelliInspect – AI-Powered Industrial Defect Detection System
Main Streamlit Application Entrypoint
"""

import os
import sys

# Configure YOLO offline & writable config dir
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["YOLO_CONFIG_DIR"] = os.path.join(PROJECT_ROOT, ".yolo")
os.environ["YOLO_OFFLINE"] = "true"
os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

import streamlit as st

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.db import init_db
from components.dashboard import render_dashboard
from components.inspection import render_inspection
from components.history import render_history
from components.performance import render_performance
from components.about import render_about


# Page Configuration
st.set_page_config(
    page_title="IntelliInspect | Industrial AI Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_custom_css():
    """Load dark navy industrial theme stylesheet."""
    css_path = os.path.join(PROJECT_ROOT, "static", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    # Initialize database tables
    init_db()

    # Load custom industrial styles
    load_custom_css()

    # Sidebar Navigation & System Telemetry
    with st.sidebar:
        st.markdown("""
            <div class="brand-badge">
                <div class="brand-icon">🔬</div>
                <div>
                    <h3 class="brand-title">IntelliInspect</h3>
                    <p class="brand-subtitle">Vision AI Quality Hub</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<p style='font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px;'>NAVIGATION</p>", unsafe_allow_html=True)

        page_selection = st.radio(
            "Navigation Menu",
            [
                "📊  Dashboard",
                "🔍  Inspect Product",
                "📜  Inspection History",
                "📈  Model Performance",
                "ℹ️  About Project"
            ],
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border-color: #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)

        # System Telemetry & Model Status Box
        weights_exist = os.path.exists(os.path.join(PROJECT_ROOT, "models", "weights", "best.pt"))
        weights_badge = "FINE-TUNED WEIGHTS" if weights_exist else "BASE YOLOV8"
        badge_color = "#38bdf8" if weights_exist else "#94a3b8"

        st.markdown(f"""
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 12px; margin-bottom: 16px;">
                <div style="font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px;">
                    SYSTEM TELEMETRY
                </div>
                <div style="font-size: 12px; color: #e2e8f0; margin-bottom: 4px;">
                    <b>Model:</b> <span style="color: {badge_color};">{weights_badge}</span>
                </div>
                <div style="font-size: 12px; color: #e2e8f0; margin-bottom: 4px;">
                    <b>Database:</b> <span style="color: #22c55e;">SQLite (Active)</span>
                </div>
                <div style="font-size: 12px; color: #e2e8f0; margin-bottom: 4px;">
                    <b>Engine:</b> PyTorch 2.15 CPU
                </div>
                <div style="font-size: 12px; color: #e2e8f0;">
                    <b>Defect Classes:</b> 6 Categories
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Quick action: Sample Dataset Generator
        with st.expander("🛠️ Developer Tools", expanded=False):
            if st.button("Generate Test Benchmark Dataset", use_container_width=True):
                with st.spinner("Generating industrial surface samples & labels..."):
                    try:
                        from dataset.download_dataset import generate_sample_dataset
                        generate_sample_dataset(num_train=30, num_val=10)
                        st.success("Samples & validation split generated!")
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.markdown("""
            <div style="color: #475569; font-size: 11px; text-align: center; margin-top: 30px;">
                IntelliInspect v1.0.0<br>
                Production Vision Quality System
            </div>
        """, unsafe_allow_html=True)

    # Route Page Components
    if "Dashboard" in page_selection:
        render_dashboard()
    elif "Inspect Product" in page_selection:
        render_inspection()
    elif "Inspection History" in page_selection:
        render_history()
    elif "Model Performance" in page_selection:
        render_performance()
    elif "About Project" in page_selection:
        render_about()


if __name__ == "__main__":
    main()
