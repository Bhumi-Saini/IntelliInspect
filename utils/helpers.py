"""
IntelliInspect - Helper Utilities
Visualization, chart generation, file handling, and formatting helpers.
"""

import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "static", "uploads")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "static", "results")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "generated_reports")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")

# Ensure required directories exist
for directory in [UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR, SAMPLES_DIR]:
    os.makedirs(directory, exist_ok=True)


def generate_inspection_id() -> str:
    """Generate a clean, professional inspection identifier."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:4].upper()
    return f"INSP-{timestamp}-{short_uuid}"


def save_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """
    Save Streamlit uploaded file into static/uploads directory.
    Returns: (saved_absolute_path, filename)
    """
    filename = f"{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath, uploaded_file.name


def save_result_image(image_pil_or_cv, prefix: str = "result") -> str:
    """Save annotated result image into static/results directory."""
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    filepath = os.path.join(RESULTS_DIR, filename)

    if isinstance(image_pil_or_cv, Image.Image):
        image_pil_or_cv.save(filepath, format="PNG")
    else:
        # Assume RGB numpy array
        img_bgr = image_pil_or_cv[:, :, ::-1]
        import cv2
        cv2.imwrite(filepath, img_bgr)

    return filepath


# ==============================================================================
# PLOTLY CHART BUILDERS FOR MODERN INDUSTRIAL DASHBOARD
# ==============================================================================

# Industrial Color Palette
COLOR_PASS = "#22c55e"      # Emerald 500
COLOR_DEFECT = "#ef4444"    # Rose 500
COLOR_REVIEW = "#f59e0b"    # Amber 500
DARK_BG = "#0f172a"         # Slate 900
DARK_CARD = "#1e293b"       # Slate 800
TEXT_MUTED = "#94a3b8"      # Slate 400


def create_status_donut_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create modern donut chart displaying breakdown of Pass vs Defect vs Review."""
    labels = ["Pass (Compliant)", "Defect Detected", "Review Required"]
    values = [
        metrics.get("passed_count", 0),
        metrics.get("defective_count", 0),
        metrics.get("review_count", 0)
    ]
    colors = [COLOR_PASS, COLOR_DEFECT, COLOR_REVIEW]

    if sum(values) == 0:
        # Empty placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="No inspection data yet",
            showarrow=False,
            font=dict(size=14, color=TEXT_MUTED)
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=10, r=10, t=20, b=20)
        )
        return fig

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="percent+value",
        hoverinfo="label+value+percent",
        textfont=dict(size=13, color="#ffffff")
    )])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#f8fafc"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color=TEXT_MUTED)
        ),
        margin=dict(l=15, r=15, t=10, b=10),
        height=280
    )
    return fig


def create_defect_bar_chart(distribution: List[Dict[str, Any]]) -> go.Figure:
    """Create horizontal bar chart of defect categories and occurrences."""
    if not distribution:
        fig = go.Figure()
        fig.add_annotation(
            text="No defect records found",
            showarrow=False,
            font=dict(size=14, color=TEXT_MUTED)
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=10, r=10, t=20, b=20)
        )
        return fig

    df = pd.DataFrame(distribution)
    df["label"] = df["defect_class"].str.replace("_", " ").str.title()
    df = df.sort_values(by="count", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["count"],
        y=df["label"],
        orientation="h",
        marker=dict(
            color=df["count"],
            colorscale=[[0, "#38bdf8"], [1, "#0284c7"]],
            line=dict(color="#38bdf8", width=1)
        ),
        text=df["count"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Detected Count: %{x}<extra></extra>"
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#f8fafc"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#334155",
            zeroline=False,
            title=dict(text="Total Occurrences", font=dict(color=TEXT_MUTED, size=12))
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        margin=dict(l=20, r=30, t=15, b=20),
        height=280
    )
    return fig


def create_confidence_distribution_chart(inspections: List[Dict[str, Any]]) -> go.Figure:
    """Create confidence distribution histogram."""
    confs = [row["avg_confidence"] * 100 for row in inspections if row.get("avg_confidence", 0) > 0]
    if not confs:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for confidence distribution",
            showarrow=False,
            font=dict(size=14, color=TEXT_MUTED)
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=10, r=10, t=20, b=20)
        )
        return fig

    fig = go.Figure(data=[go.Histogram(
        x=confs,
        nbinsx=15,
        marker_color="#6366f1",
        opacity=0.85,
        marker=dict(line=dict(color="#818cf8", width=1))
    )])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#f8fafc"),
        xaxis=dict(
            title=dict(text="Model Confidence (%)", font=dict(color=TEXT_MUTED, size=12)),
            showgrid=True,
            gridcolor="#334155"
        ),
        yaxis=dict(
            title=dict(text="Frequency", font=dict(color=TEXT_MUTED, size=12)),
            showgrid=True,
            gridcolor="#334155"
        ),
        margin=dict(l=20, r=20, t=15, b=20),
        height=280
    )
    return fig


def create_timeline_trend_chart(inspections: List[Dict[str, Any]]) -> go.Figure:
    """Create trend line chart over recent inspections."""
    if len(inspections) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 inspection records for trend analysis",
            showarrow=False,
            font=dict(size=14, color=TEXT_MUTED)
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=10, r=10, t=20, b=20)
        )
        return fig

    df = pd.DataFrame(inspections).iloc[::-1]  # chronological order
    df["defect_binary"] = df["status"].apply(lambda s: 1 if s == "DEFECT DETECTED" else 0)
    df["rolling_defect_rate"] = df["defect_binary"].rolling(window=5, min_periods=1).mean() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["rolling_defect_rate"],
        mode="lines+markers",
        line=dict(color="#f43f5e", width=2.5),
        marker=dict(size=6, color="#fda4af"),
        name="Defect Rate % (5-sample Moving Avg)"
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#f8fafc"),
        xaxis=dict(
            title=dict(text="Inspection Timestamp", font=dict(color=TEXT_MUTED, size=12)),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="Defect Rate (%)", font=dict(color=TEXT_MUTED, size=12)),
            showgrid=True,
            gridcolor="#334155",
            range=[0, 105]
        ),
        margin=dict(l=20, r=20, t=15, b=20),
        height=280
    )
    return fig
