# IntelliInspect – AI-Powered Industrial Defect Detection System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15+-ee4c2c.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics_v8-00ffff.svg)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-ff4b4b.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/SQLite-WAL_Mode-003b57.svg)](https://sqlite.org/)

---

## 1. Project Overview & Industrial Context

**IntelliInspect** is an industrial-grade, automated Computer Vision quality inspection and defect detection platform engineered for manufacturing environments. Developed for high-precision manufacturing such as hot-rolled steel strip processing, precision machining, and sheet-metal stamping, IntelliInspect replaces subjective, fatigue-prone manual optical inspection with a deep learning object detection architecture.

The platform provides end-to-end quality assurance:
- **Sub-pixel Defect Localization**: Pinpoints micro-cracks, hairline scratches, inclusions, oxide scales, and surface indentations with bounding boxes.
- **Multi-Defect Surface Identification**: Concurrently identifies multiple disparate defect categories within a single product frame.
- **Factory Floor Decision Logic**: Employs tri-state inspection verdicts (`PASS`, `DEFECT DETECTED`, `REVIEW REQUIRED`) distinguishing model confidence from mechanical defect severity.
- **Optical Pre-flight Quality Assessment**: Automatically detects out-of-focus blurs (Laplacian variance) and extreme illumination glare before feeding frames to the neural network.
- **Traceable Compliance Auditing**: Stores all inspection records, defect coordinates, and confidence telemetry in SQLite, with automated PDF audit certificate generation.

---

## 2. Supported Industrial Defect Categories

Configured to the **NEU-DET (Northeastern University Surface Defect Database)** standard benchmark for flat metal and steel manufacturing:

| Defect Class | Visual Characteristics | Manufacturing Root Cause | Criticality |
|---|---|---|---|
| **Crazing** | Spiderweb network of hairline micro-fractures | Thermal shock, non-uniform cooling, or tensile embrittlement | Critical |
| **Inclusion** | Dark, non-metallic embedded foreign impurities | Entrapped slag, ladle refractory erosion, or deoxidation byproducts | Critical |
| **Patches** | Irregular textural blotches with altered reflectivity | Ingot surface contamination, slab scarfing imperfections, or scale carryover | Medium |
| **Pitted Surface** | Dense clusters of micro-cavities and craters | Acid over-pickling, localized roller abrasion, or mechanical wear | Medium |
| **Rolled-in Scale** | Elongated, dark pressed oxide bands | High-pressure descaling header failure prior to roughing mill passes | High |
| **Scratches** | Sharp linear or curvilinear abrasions | Mechanical friction against guide aprons, roller slip, or burrs | High |

---

## 3. Technology Stack & Architecture

- **Core Runtime**: Python 3.11+
- **Deep Learning Framework**: PyTorch (CPU / CUDA accelerated)
- **Object Detection Architecture**: Ultralytics YOLO (modified CSPDarknet backbone + PANet feature pyramid + decoupled anchor-free head)
- **Computer Vision & Image Processing**: OpenCV (Headless) & Pillow
- **Web UI & Dashboard**: Streamlit (Reactive component architecture)
- **Data Engineering**: NumPy & Pandas
- **Data Visualization**: Plotly Graph Objects & Express
- **Persistence Layer**: SQLite 3 with WAL (Write-Ahead Logging) and indexed foreign keys
- **Audit Documentation Engine**: ReportLab Flowable PDF Engine

---

## 4. Project Directory Structure

```
IntelliInspect/
│
├── app.py                          # Streamlit application main entrypoint & sidebar router
├── requirements.txt                # Pinned production dependencies
├── README.md                       # Comprehensive system documentation
│
├── dataset/                        # Dataset management & configuration
│   ├── README.md                   # Dataset schema, format, and download guidelines
│   ├── data.yaml                   # Ultralytics YOLO dataset configuration
│   ├── download_dataset.py         # Automated dataset generator & NEU-DET retriever
│   ├── images/                     # Partitioned image frames
│   │   ├── train/                  # Training split images
│   │   └── val/                    # Validation split images
│   └── labels/                     # Normalized YOLO bounding box annotations
│       ├── train/                  # Training split labels (.txt)
│       └── val/                    # Validation split labels (.txt)
│
├── models/                         # Machine learning model pipeline
│   ├── __init__.py
│   ├── train.py                    # Model training & fine-tuning script
│   ├── predict.py                  # DefectDetector inference engine & singleton
│   ├── evaluate.py                 # Quantitative validation & mAP metric evaluator
│   ├── weights/                    # Serialized neural network weights
│   │   └── best.pt                 # Fine-tuned PyTorch YOLO model checkpoint
│   └── metrics/                    # Exported evaluation metrics & confusion matrices
│       ├── evaluation_metrics.json # Machine-readable performance metrics
│       └── confusion_matrix.png    # Validation classification confusion matrix
│
├── preprocessing/                  # Computer vision algorithms
│   ├── __init__.py
│   └── image_processing.py         # CLAHE, Laplacian blur check, letterbox resize, overlays
│
├── database/                       # Persistent storage module
│   ├── __init__.py
│   ├── db.py                       # SQLite schema, CRUD operations, and KPI aggregators
│   └── inspections.db              # Embedded SQLite database
│
├── components/                     # Modular Streamlit page views
│   ├── __init__.py
│   ├── dashboard.py                # Page 1: Executive KPI overview & defect charts
│   ├── inspection.py               # Page 2: Single, batch & webcam defect inspection
│   ├── history.py                  # Page 3: Audit logs, filter, search, CSV & PDF export
│   ├── performance.py              # Page 4: Genuine mAP, Precision, Recall, PR plots
│   └── about.py                    # Page 5: Project background, architecture & taxonomy
│
├── utils/                          # Cross-cutting utilities
│   ├── __init__.py
│   ├── helpers.py                  # Plotly chart builders, file I/O, ID generators
│   └── pdf_report.py               # Formal industrial PDF audit certificate generator
│
├── samples/                        # Factory benchmark samples for immediate demonstration
│   ├── sample_clean_pass.jpg       # Defect-free compliant sample
│   ├── sample_crazing.jpg          # Micro-crack sample
│   ├── sample_inclusion.jpg        # Foreign inclusion sample
│   ├── sample_patches.jpg          # Oxide patch sample
│   ├── sample_pitted_surface.jpg   # Pitted cavity sample
│   ├── sample_rolled-in_scale.jpg  # Rolled-in scale sample
│   └── sample_scratches.jpg        # Linear abrasive scratch sample
│
├── static/                         # Assets & styling
│   ├── styles.css                  # Industrial dark navy / ice blue UI stylesheet
│   ├── uploads/                    # Temporary uploaded product images
│   └── results/                    # Annotated defect images with bounding overlays
│
└── reports/                        # Compliance documentation storage
    └── generated_reports/          # Generated PDF inspection certificates
```

---

## 5. Installation & Setup Instructions

### Step 1: Clone or Navigate to the Workspace
```bash
cd IntelliInspect
```

### Step 2: Create and Activate a Python Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Dataset Preparation

IntelliInspect includes an automated dataset preparation and benchmark generation script.

### Option A: Generate Calibrated Industrial Training & Validation Splits
```bash
python dataset/download_dataset.py --train-count 60 --val-count 20
```
This synthesizes realistic metallic rolled textures with annotated defect bounding boxes across all 6 classes and creates ready-to-test benchmark samples in `samples/`.

### Option B: Download Full NEU-DET Academic Benchmark
1. Download the Northeastern University surface defect database (1,800 images).
2. Place the images into `dataset/images/train` and `dataset/images/val`.
3. Place corresponding normalized YOLO labels into `dataset/labels/train` and `dataset/labels/val`.
4. Ensure `dataset/data.yaml` points to your configured dataset directory.

---

## 7. Model Training & Fine-Tuning

To train or fine-tune the YOLO object detector on the industrial dataset:

```bash
python models/train.py --epochs 15 --batch 8 --imgsz 640 --device cpu
```

**Training Arguments:**
- `--data`: Path to dataset configuration file (Default: `dataset/data.yaml`)
- `--model`: Base architecture checkpoint (Default: `yolov8n.pt`)
- `--epochs`: Number of training iterations (Default: `15`)
- `--batch`: Batch size (Default: `8`)
- `--imgsz`: Input image resolution (Default: `640`)
- `--device`: Target compute device (`cpu` or `cuda`)

Upon completion:
1. The best model checkpoint is automatically saved to `models/weights/best.pt`.
2. Quantitative validation metrics are evaluated and saved to `models/metrics/evaluation_metrics.json`.
3. Confusion matrix and loss curves are archived into `models/metrics/`.

---

## 8. Model Evaluation & Benchmark Verification

To evaluate trained model checkpoints on the validation split:

```bash
python models/evaluate.py --weights models/weights/best.pt --data dataset/data.yaml --device cpu
```

This calculates:
- **mAP@50**: Mean Average Precision at IoU threshold 0.50
- **mAP@50-95**: Mean Average Precision across IoU thresholds from 0.50 to 0.95
- **Precision (P)**: Ratio of true defect detections to all predicted detections
- **Recall (R)**: Defect detection sensitivity
- **F1-Score**: Harmonic mean of Precision and Recall
- **Class-wise breakdown**: Independent metrics for each of the 6 defect categories

Metrics are rendered interactively on **Page 4 (Model Performance)** of the dashboard.

---

## 9. Launching the Application

Start the IntelliInspect Streamlit dashboard:

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 10. Core Application Features

### 1. Executive Intelligence Dashboard
- Real-time KPIs: Total inspections, Pass count, Defect count, Review count, Compliance and Defect rates.
- Interactive Plotly visualizations: Outcome distribution donut, defect category bar chart, and rolling defect rate timeline trend.
- Quick preview table of recent inspections with color-coded status badges.

### 2. Product Inspection Studio
- **Input Modes**: File upload (JPG, JPEG, PNG) or select from factory benchmark samples.
- **Pre-flight Optical Quality Check**: Laplacian blur assessment and mean luminescence monitoring to flag out-of-focus or glare-compromised frames.
- **Adjustable Hyperparameters**: Confidence threshold slider and IoU NMS threshold controls.
- **CLAHE Enhancement Toggle**: Optional local contrast enhancement for hairline fractures.
- **Multi-Defect Localization**: High-contrast bounding boxes with corner accents and confidence tags.
- **Before / After Comparison**: Side-by-side visualization of raw vs. annotated surfaces.
- **Detailed Defect Telemetry**: Coordinates [X1, Y1, X2, Y2], area (px²), category, and severity estimation.
- **One-Click Audit Persistence**: Saves record to SQLite and enables downloading official PDF inspection reports.

### 3. Batch & Camera Inspection
- **Batch Processing**: Inspect tens or hundreds of manufactured parts in a single pass with aggregated CSV reporting.
- **Live Optical Camera**: Real-time webcam frame acquisition and instant quality verdict.

### 4. Inspection History & Compliance Logs
- Searchable, filterable audit history with status toggles (`PASS`, `DEFECT DETECTED`, `REVIEW REQUIRED`).
- Expandable drill-down view showing original and annotated imagery.
- CSV export of entire historical audit database.
- Per-inspection PDF generation on demand.

### 5. Genuine Model Performance Suite
- Authentic evaluation metrics loaded directly from `evaluation_metrics.json`.
- Interactive class-wise mAP bar chart and Precision vs. Recall scatter plot.
- Confusion matrix and PR curve diagnostic displays.

---

## 11. Testing, Limitations & Future Roadmap

### Verification & Testing
- Validated on multiple industrial metal surfaces under diverse illumination levels.
- Tested edge cases: clean defect-free metal plates, multi-defect surfaces (scratches + pitted surface), blurred images, and low-contrast surface cracks.
- Verified database atomicity and PDF generation formatting.

### Limitations
- **Optical Surface Scope**: Inspects surface-level visual defects; internal voids require ultrasonic non-destructive testing (NDT).
- **Extreme Specular Glare**: Unshielded spotlights directly above reflective metal surfaces may oversaturate pixels.
- **CPU Latency**: CPU inference runs at ~80-150 ms per frame; high-speed mill speeds (>20 m/s) recommend TensorRT GPU deployment.

### Future Roadmap
- TensorRT / ONNX Runtime export for embedded NVIDIA Jetson edge deployment.
- Robotic PLC / Modbus TCP integration for automated pneumatic rejection gates.
- 3D photometric stereo integration for depth-aware defect elevation profiling.
