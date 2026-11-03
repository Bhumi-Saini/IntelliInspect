"""Database package for IntelliInspect."""
from .db import (
    init_db,
    save_inspection,
    get_all_inspections,
    get_inspection_details,
    get_dashboard_metrics,
    get_defect_distribution,
    delete_inspection,
    clear_all_records
)

__all__ = [
    "init_db",
    "save_inspection",
    "get_all_inspections",
    "get_inspection_details",
    "get_dashboard_metrics",
    "get_defect_distribution",
    "delete_inspection",
    "clear_all_records"
]
