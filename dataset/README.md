# IntelliInspect - Industrial Defect Dataset Guide

## 1. Overview
This dataset module is configured for the **NEU-DET (Northeastern University Surface Defect Database)** and custom industrial manufacturing surface defect inspection.
The dataset focuses on surface defects on hot-rolled steel strip products, providing bounding box annotations formatted for YOLO object detection.

## 2. Defect Classes
The system identifies 6 primary industrial defect categories:
1. **crazing** (Class ID: 0) - Spiderweb-like network of surface micro-cracks caused by thermal stresses or tensile embrittlement.
2. **inclusion** (Class ID: 1) - Non-metallic embedded impurities (slag, deoxidation byproducts, refractory remnants).
3. **patches** (Class ID: 2) - Irregular textural blotches and uneven oxide scale accumulation.
4. **pitted_surface** (Class ID: 3) - Microscopic indentations, cavities, and crater-like pitting from roller wear or acid over-pickling.
5. **rolled-in_scale** (Class ID: 4) - Dark pressed oxide scale embedded into the strip surface during hot rolling.
6. **scratches** (Class ID: 5) - Linear or curvilinear abrasive scores from mechanical friction or roller misalignment.

## 3. Directory Structure
```
dataset/
├── data.yaml              # Ultralytics YOLO dataset configuration
├── download_dataset.py    # Automated dataset retriever & synthetic generator
├── images/
│   ├── train/             # Training images (.jpg/.png)
│   └── val/               # Validation images (.jpg/.png)
└── labels/
    ├── train/             # YOLO format labels (.txt)
    └── val/               # YOLO format labels (.txt)
```

## 4. YOLO Label Format
Each line in a `.txt` label file represents one defect bounding box:
`<class_id> <x_center> <y_center> <width> <height>`

Coordinates are normalized relative to image dimensions:
- `x_center`, `y_center`, `width`, `height` are floating-point numbers between `0.0` and `1.0`.

## 5. Dataset Setup Instructions
You can prepare the dataset in two ways:
1. **Automated Setup / Quick Start**:
   Run `python dataset/download_dataset.py` to generate and verify baseline training and validation images with annotations.
2. **Full NEU-DET Dataset**:
   Download the original NEU-DET dataset from Kaggle or Northeastern University repository:
   - Extract the 1,800 grayscale images (200x200 px).
   - Convert XML annotations to YOLO format using `download_dataset.py --convert-xml <path_to_xml>`.
   - Organize into `images/train`, `images/val`, `labels/train`, `labels/val`.
