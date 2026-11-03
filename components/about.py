"""
IntelliInspect - About Project Component
System documentation, architecture explanation, defect taxonomy, and machine learning methodology.
"""

import streamlit as st


def render_about():
    """Render comprehensive project background, architecture, and methodology."""
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="margin: 0; font-size: 26px; font-weight: 700; color: #f8fafc;">
                About IntelliInspect – AI-Powered Quality Inspection
            </h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">
                Deep learning computer vision architecture for automated surface defect detection in advanced manufacturing.
            </p>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Project Overview & Objectives",
        "Machine Learning Pipeline",
        "Defect Taxonomy (NEU-DET)",
        "System Architecture & Limitations"
    ])

    with tab1:
        st.markdown("""
        ### Executive Summary
        **IntelliInspect** is an end-to-end industrial computer vision defect detection and automated quality control system.
        Manufacturing industries—specifically flat steel, hot-rolled coils, precision automotive parts, and semiconductor wafer fabrication—face
        strict tolerances where microscopic or macro surface irregularities cause catastrophic mechanical failure, structural fatigue, or severe financial waste.

        ### Problem Statement
        Traditional manual optical inspection (AOI) relying on human operators suffers from:
        - **Operator Fatigue**: Visual acuity drops by over 40% after just 45 minutes of continuous visual scanning.
        - **Subjectivity & Variability**: Inconsistent defect classification criteria between inspection shifts and plants.
        - **Production Bottlenecks**: High-speed rolling mills (operating at 10-25 m/s) cannot be inspected manually in real-time.
        - **High Latency & Re-inspection Costs**: Delayed identification leads to thousands of meters of defective scrap produced before line intervention.

        ### Engineering Objectives
        1. **Automated Defect Localization & Classification**: Detect bounding boxes and categorize defects (crazing, patches, pitted surfaces, inclusion, rolled-in scale, scratches).
        2. **Multi-Defect Support**: Detect multiple defect categories simultaneously within complex product surfaces.
        3. **Real-time Inference**: Achieve sub-second inference speeds suitable for edge and near-line deployment.
        4. **Transparent Quality Criteria**: Clear separation of *model confidence* from *defect severity*, preventing false passes.
        5. **Traceable Quality Audits**: Maintain persistent SQLite audit history with downloadable PDF compliance documentation.
        """)

    with tab2:
        st.markdown("""
        ### Deep Learning Methodology & Model Architecture

        #### 1. Model Backbone: YOLO (You Only Look Once)
        IntelliInspect leverages a fine-tuned **YOLO** (Ultralytics) single-stage detector optimized for industrial edge inference:
        - **Backbone (Modified CSPDarknet / C2f modules)**: Extracts hierarchical spatial and semantic features across high, medium, and low receptive fields.
        - **Neck (PANet / Path Aggregation Network)**: Enhances feature propagation across scales, critical for detecting fine hairline scratches and micro-cracks alongside larger rolled-in scale patches.
        - **Head**: Decoupled anchor-free detection head predicting bounding box coordinates, objectness, and class probability distributions.

        #### 2. Loss Formulations
        The loss function balances three distinct objectives:
        - **Bounding Box Regression**: Complete Intersection over Union (CIoU) / Distribution Focal Loss (DFL) for sub-pixel boundary localization.
        - **Classification Loss**: Binary Cross-Entropy (BCE) with Focal Loss weighting to counteract class imbalance.
        - **Feature-level Regularization**: Weight decay and stochastic gradient descent with momentum.

        #### 3. Image Preprocessing & Optical Enhancement
        - **Aspect-ratio Preserving Letterbox**: Standardizes inputs to 640x640 resolution without aspect distortion.
        - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Equalizes localized luminescence in the LAB color space to extract faint texture anomalies under uneven factory illumination.
        - **Optical Quality Guard**: Laplacian variance check flags out-of-focus and blurred frames, routing them to `REVIEW REQUIRED` instead of generating erroneous passes.
        """)

    with tab3:
        st.markdown("""
        ### Supported Industrial Defect Categories (NEU-DET Benchmark)

        The Northeastern University (NEU) surface defect database contains 6 distinct surface defect classes typically found on hot-rolled steel strips:

        | Defect Category | Visual Characteristics | Manufacturing Root Cause | Severity Level |
        |---|---|---|---|
        | **Scratches** | Sharp linear or curvilinear abrasions along rolling direction | Mechanical friction, guide plate misalignment, or debris in pinch rolls | High |
        | **Cracks / Crazing** | Fine spiderweb network of micro-fractures | Thermal shock, uneven cooling rates, or tensile embrittlement | Critical |
        | **Rolled-in Scale** | Dark, irregular, pressed oxide patches | Failure of high-pressure descaling jets before rolling pass | High |
        | **Patches** | Localized blotches of irregular texture and oxidation | Ingot surface contamination, slab scarfing defects, or slag carryover | Medium |
        | **Pitted Surface** | Dense clusters of small cavities and crater-like depressions | Acid over-pickling, localized roll wear, or corrosion pitting | Medium |
        | **Inclusion** | Non-metallic foreign material embedded in the metal matrix | Deoxidation products, mold powder entrapment, or ladle refractory erosion | Critical |
        """)

    with tab4:
        st.markdown("""
        ### Complete Technology Stack

        - **Core Runtime**: Python 3.11
        - **Deep Learning Engine**: PyTorch & Ultralytics YOLOv8
        - **Computer Vision**: OpenCV (Headless) & Pillow
        - **Front-End Dashboard**: Streamlit (Reactive Web App)
        - **Data Handling**: NumPy & Pandas
        - **Visual Analytics**: Plotly Interactive Graphics
        - **Audit Database**: SQLite 3 with WAL Mode and Indexing
        - **Report Engine**: ReportLab PDF Engine

        ### Current System Limitations & Boundary Conditions
        1. **Lighting Uniformity**: Extreme specular glare from uncalibrated factory spotlights may obscure surface details.
        2. **Orientation Invariance**: Model is trained on canonical surface orientations; severe oblique camera angles (>45 degrees) reduce recall.
        3. **Sub-surface Defects**: Optical cameras only register surface-manifested defects; internal voids require ultrasonic or radiographic testing (NDT).
        4. **Compute Hardware**: Real-time throughput (30+ FPS) requires GPU acceleration (NVIDIA TensorRT / CUDA); CPU inference runs at ~5-15 FPS.
        """)
