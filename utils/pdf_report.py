"""
IntelliInspect - PDF Inspection Report Generator
Generates formal industrial audit reports with defect bounding boxes, metrics, and quality sign-off.
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def generate_inspection_pdf(
    inspection_data: Dict[str, Any],
    defect_items: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    Generate a comprehensive PDF audit report for an inspection.
    Returns: output_path
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=12
    )

    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6
    )

    normal_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155")
    )

    story = []

    # Header / Title
    story.append(Paragraph("INTELLIINSPECT | INDUSTRIAL QUALITY AUDIT REPORT", title_style))
    story.append(Paragraph("Automated Deep Learning Vision Inspection System & Defect Verification", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=14))

    # Inspection Metadata Table
    status = inspection_data.get("status", "UNKNOWN")
    if status == "PASS":
        status_color = colors.HexColor("#16a34a")
        status_bg = colors.HexColor("#dcfce7")
    elif status == "DEFECT DETECTED":
        status_color = colors.HexColor("#dc2626")
        status_bg = colors.HexColor("#fee2e2")
    else:
        status_color = colors.HexColor("#d97706")
        status_bg = colors.HexColor("#fef3c7")

    meta_data = [
        [
            Paragraph("<b>Inspection ID:</b>", normal_style),
            Paragraph(str(inspection_data.get("inspection_id", "N/A")), normal_style),
            Paragraph("<b>Audit Date / Time:</b>", normal_style),
            Paragraph(str(inspection_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))), normal_style)
        ],
        [
            Paragraph("<b>Sample Image:</b>", normal_style),
            Paragraph(str(inspection_data.get("image_name", "N/A")), normal_style),
            Paragraph("<b>Audit Verdict:</b>", normal_style),
            Paragraph(f"<font color='{status_color.hexval()}'><b>{status}</b></font>", normal_style)
        ],
        [
            Paragraph("<b>Defects Count:</b>", normal_style),
            Paragraph(str(inspection_data.get("total_defects", 0)), normal_style),
            Paragraph("<b>Defect Categories:</b>", normal_style),
            Paragraph(str(inspection_data.get("defect_categories", "None")), normal_style)
        ],
        [
            Paragraph("<b>Average Confidence:</b>", normal_style),
            Paragraph(f"{float(inspection_data.get('avg_confidence', 0.0)):.1%}", normal_style),
            Paragraph("<b>Max Confidence:</b>", normal_style),
            Paragraph(f"{float(inspection_data.get('max_confidence', 0.0)):.1%}", normal_style)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[1.5 * inch, 2.0 * inch, 1.5 * inch, 2.5 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Visual Evidence
    res_img_path = inspection_data.get("result_image_path")
    if res_img_path and os.path.exists(res_img_path):
        story.append(Paragraph("Visual Evidence & Defect Localization", h2_style))
        try:
            # Scale image proportionally to fit page width
            img_rl = RLImage(res_img_path, width=4.5 * inch, height=3.0 * inch)
            story.append(img_rl)
            story.append(Spacer(1, 12))
        except Exception as e:
            story.append(Paragraph(f"Image display error: {e}", normal_style))

    # Defect Items Table
    if defect_items:
        story.append(Paragraph("Detected Defect Catalog & Bounding Coordinates", h2_style))
        item_rows = [
            [
                Paragraph("<b>#</b>", normal_style),
                Paragraph("<b>Defect Classification</b>", normal_style),
                Paragraph("<b>Confidence</b>", normal_style),
                Paragraph("<b>Bounding Box (X1, Y1, X2, Y2)</b>", normal_style)
            ]
        ]

        for i, item in enumerate(defect_items, 1):
            bbox_str = f"[{item.get('bbox_x1', 0):.0f}, {item.get('bbox_y1', 0):.0f}, {item.get('bbox_x2', 0):.0f}, {item.get('bbox_y2', 0):.0f}]"
            item_rows.append([
                Paragraph(str(i), normal_style),
                Paragraph(f"<b>{item.get('defect_class', 'Unknown').upper()}</b>", normal_style),
                Paragraph(f"{float(item.get('confidence', 0.0)):.1%}", normal_style),
                Paragraph(bbox_str, normal_style)
            ])

        items_table = Table(item_rows, colWidths=[0.5 * inch, 2.5 * inch, 1.5 * inch, 3.0 * inch])
        items_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
        ]))
        story.append(items_table)
        story.append(Spacer(1, 14))

    # Notes & Sign-off
    story.append(Spacer(1, 10))
    story.append(Paragraph("Inspection Notes & Certification", h2_style))
    notes_text = inspection_data.get("notes") or "Automated optical inspection completed successfully via IntelliInspect deep vision pipeline."
    story.append(Paragraph(f"<i>Remarks:</i> {notes_text}", normal_style))
    story.append(Spacer(1, 15))

    sign_data = [
        [
            Paragraph("<b>Quality Assurance Lead:</b> ____________________", normal_style),
            Paragraph("<b>Automated System Signature:</b> VERIFIED", normal_style)
        ],
        [
            Paragraph("<b>Production Line Stamp:</b> LINE-QC-04", normal_style),
            Paragraph(f"<b>Report Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[4.0 * inch, 3.5 * inch])
    sign_table.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4)
    ]))
    story.append(sign_table)

    doc.build(story)
    return output_path
