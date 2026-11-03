"""
IntelliInspect - Dashboard Component
Executive analytics overview, KPI metrics, defect distribution, and recent activity.
"""

import streamlit as st
import pandas as pd
from database.db import get_dashboard_metrics, get_defect_distribution, get_all_inspections
from utils.helpers import (
    create_status_donut_chart,
    create_defect_bar_chart,
    create_timeline_trend_chart
)


def render_dashboard():
    """Render Executive Quality Analytics Dashboard."""
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="margin: 0; font-size: 26px; font-weight: 700; color: #f8fafc;">
                Factory Inspection Intelligence Dashboard
            </h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">
                Real-time quality control telemetry, defect distribution, and statistical process monitoring.
            </p>
        </div>
    """, unsafe_allow_html=True)

    metrics = get_dashboard_metrics()
    defect_dist = get_defect_distribution()
    recent_inspections = get_all_inspections(limit=10)

    # 4 Primary KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TOTAL INSPECTIONS</div>
                <div class="metric-value">{metrics['total_inspections']:,}</div>
                <div class="metric-subtext">Processed product frames</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PASSED (COMPLIANT)</div>
                <div class="metric-value" style="color: #22c55e;">{metrics['passed_count']:,}</div>
                <div class="metric-subtext">{metrics['pass_rate']}% compliance rate</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">DEFECTS DETECTED</div>
                <div class="metric-value" style="color: #ef4444;">{metrics['defective_count']:,}</div>
                <div class="metric-subtext">{metrics['defect_rate']}% rejection rate</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">REVIEW REQUIRED</div>
                <div class="metric-value" style="color: #f59e0b;">{metrics['review_count']:,}</div>
                <div class="metric-subtext">Low conf / image blur check</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 2 Main Visual Analytics Charts
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.markdown("""
            <div class="panel-box">
                <h4 style="margin: 0 0 12px 0; font-size: 16px; color: #f1f5f9;">
                    Inspection Outcome Distribution
                </h4>
            </div>
        """, unsafe_allow_html=True)
        donut_fig = create_status_donut_chart(metrics)
        st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        st.markdown("""
            <div class="panel-box">
                <h4 style="margin: 0 0 12px 0; font-size: 16px; color: #f1f5f9;">
                    Identified Defect Class Breakdown
                </h4>
            </div>
        """, unsafe_allow_html=True)
        bar_fig = create_defect_bar_chart(defect_dist)
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    # Historical Defect Rate Trend
    st.markdown("""
        <div class="panel-box" style="margin-top: 15px;">
            <h4 style="margin: 0 0 10px 0; font-size: 16px; color: #f1f5f9;">
                Defect Rate Temporal Trend (Moving Average)
            </h4>
        </div>
    """, unsafe_allow_html=True)
    all_recent = get_all_inspections(limit=50)
    trend_fig = create_timeline_trend_chart(all_recent)
    st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})

    # Recent Inspection Table
    st.markdown("""
        <div style="margin-top: 24px; margin-bottom: 12px;">
            <h4 style="margin: 0; font-size: 18px; color: #f1f5f9;">Recent Quality Inspections</h4>
        </div>
    """, unsafe_allow_html=True)

    if not recent_inspections:
        st.info("No inspection records available in SQLite database. Navigate to 'Inspect Product' to run the first inspection.")
    else:
        display_data = []
        for r in recent_inspections:
            display_data.append({
                "Inspection ID": r["inspection_id"],
                "Timestamp": r["timestamp"],
                "Image Name": r["image_name"],
                "Total Defects": r["total_defects"],
                "Categories": r["defect_categories"],
                "Avg Conf": f"{float(r['avg_confidence']):.1%}" if r["avg_confidence"] > 0 else "N/A",
                "Status": r["status"]
            })

        df_display = pd.DataFrame(display_data)

        # Style status column
        def color_status(val):
            if val == "PASS":
                return "color: #22c55e; font-weight: bold;"
            elif val == "DEFECT DETECTED":
                return "color: #ef4444; font-weight: bold;"
            elif val == "REVIEW REQUIRED":
                return "color: #f59e0b; font-weight: bold;"
            return ""

        if hasattr(df_display.style, "map"):
            styled_df = df_display.style.map(color_status, subset=["Status"])
        else:
            styled_df = df_display.style.applymap(color_status, subset=["Status"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
