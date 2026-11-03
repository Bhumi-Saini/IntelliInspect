"""
IntelliInspect - Database Operations Module
SQLite database management for inspection history, defect records, and analytics.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "database",
    "inspections.db"
)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get SQLite database connection with row factory enabled."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables and indexes."""
    conn = get_connection(db_path)
    try:
        with conn:
            # Inspections Master Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    image_path TEXT,
                    result_image_path TEXT,
                    total_defects INTEGER NOT NULL DEFAULT 0,
                    defect_categories TEXT NOT NULL DEFAULT 'None',
                    avg_confidence REAL NOT NULL DEFAULT 0.0,
                    max_confidence REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL CHECK(status IN ('PASS', 'DEFECT DETECTED', 'REVIEW REQUIRED')),
                    notes TEXT
                )
            """)

            # Individual Defect Detections Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS defect_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id TEXT NOT NULL,
                    defect_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x1 REAL NOT NULL,
                    bbox_y1 REAL NOT NULL,
                    bbox_x2 REAL NOT NULL,
                    bbox_y2 REAL NOT NULL,
                    FOREIGN KEY (inspection_id) REFERENCES inspections (inspection_id) ON DELETE CASCADE
                )
            """)

            # Indexes for faster queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_inspection_id ON inspections(inspection_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON inspections(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON inspections(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_defect_item_insp ON defect_items(inspection_id)")
    finally:
        conn.close()


def save_inspection(
    inspection_id: str,
    image_name: str,
    image_path: str,
    result_image_path: str,
    total_defects: int,
    defect_categories: List[str],
    avg_confidence: float,
    max_confidence: float,
    status: str,
    defect_items: Optional[List[Dict[str, Any]]] = None,
    notes: str = "",
    db_path: Optional[str] = None
) -> bool:
    """
    Save inspection results and individual defect items into the database.
    """
    conn = get_connection(db_path)
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cat_str = ", ".join(defect_categories) if defect_categories else "None"

    try:
        with conn:
            conn.execute("""
                INSERT INTO inspections (
                    inspection_id, timestamp, image_name, image_path, result_image_path,
                    total_defects, defect_categories, avg_confidence, max_confidence, status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inspection_id, now_iso, image_name, image_path, result_image_path,
                total_defects, cat_str, float(avg_confidence), float(max_confidence), status, notes
            ))

            if defect_items:
                items_data = [
                    (
                        inspection_id,
                        item.get("class", "Unknown"),
                        float(item.get("confidence", 0.0)),
                        float(item.get("bbox", [0, 0, 0, 0])[0]),
                        float(item.get("bbox", [0, 0, 0, 0])[1]),
                        float(item.get("bbox", [0, 0, 0, 0])[2]),
                        float(item.get("bbox", [0, 0, 0, 0])[3])
                    )
                    for item in defect_items
                ]
                conn.executemany("""
                    INSERT INTO defect_items (
                        inspection_id, defect_class, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, items_data)
        return True
    except Exception as e:
        print(f"Database error in save_inspection: {e}")
        return False
    finally:
        conn.close()


def get_all_inspections(
    status_filter: Optional[str] = None,
    search_query: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve filtered list of inspections ordered by timestamp descending."""
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM inspections WHERE 1=1"
        params: List[Any] = []

        if status_filter and status_filter != "ALL":
            query += " AND status = ?"
            params.append(status_filter)

        if search_query:
            query += " AND (inspection_id LIKE ? OR image_name LIKE ? OR defect_categories LIKE ?)"
            wildcard = f"%{search_query}%"
            params.extend([wildcard, wildcard, wildcard])

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_inspection_details(inspection_id: str, db_path: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Retrieve detailed inspection record with all detected defect items."""
    conn = get_connection(db_path)
    try:
        insp_row = conn.execute(
            "SELECT * FROM inspections WHERE inspection_id = ?", (inspection_id,)
        ).fetchone()

        if not insp_row:
            return None, []

        items_cursor = conn.execute(
            "SELECT * FROM defect_items WHERE inspection_id = ?", (inspection_id,)
        )
        items = [dict(r) for r in items_cursor.fetchall()]
        return dict(insp_row), items
    finally:
        conn.close()


def get_dashboard_metrics(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Calculate aggregate metrics for the executive overview dashboard."""
    conn = get_connection(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        passed = conn.execute("SELECT COUNT(*) FROM inspections WHERE status = 'PASS'").fetchone()[0]
        defective = conn.execute("SELECT COUNT(*) FROM inspections WHERE status = 'DEFECT DETECTED'").fetchone()[0]
        review = conn.execute("SELECT COUNT(*) FROM inspections WHERE status = 'REVIEW REQUIRED'").fetchone()[0]

        defect_rate = (defective / total * 100.0) if total > 0 else 0.0
        pass_rate = (passed / total * 100.0) if total > 0 else 0.0

        avg_conf_row = conn.execute("SELECT AVG(avg_confidence) FROM inspections WHERE avg_confidence > 0").fetchone()
        avg_confidence = avg_conf_row[0] if (avg_conf_row and avg_conf_row[0] is not None) else 0.0

        return {
            "total_inspections": total,
            "passed_count": passed,
            "defective_count": defective,
            "review_count": review,
            "defect_rate": round(defect_rate, 2),
            "pass_rate": round(pass_rate, 2),
            "avg_confidence": round(avg_confidence, 2)
        }
    finally:
        conn.close()


def get_defect_distribution(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get aggregated count and average confidence grouped by defect class."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("""
            SELECT defect_class, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM defect_items
            GROUP BY defect_class
            ORDER BY count DESC
        """)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def delete_inspection(inspection_id: str, db_path: Optional[str] = None) -> bool:
    """Delete an inspection record and its associated defect items."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM defect_items WHERE inspection_id = ?", (inspection_id,))
            conn.execute("DELETE FROM inspections WHERE inspection_id = ?", (inspection_id,))
        return True
    except Exception as e:
        print(f"Error deleting inspection: {e}")
        return False
    finally:
        conn.close()


def clear_all_records(db_path: Optional[str] = None) -> bool:
    """Clear all records from database."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM defect_items")
            conn.execute("DELETE FROM inspections")
        return True
    except Exception as e:
        print(f"Error clearing records: {e}")
        return False
    finally:
        conn.close()


# Auto-initialize database on import
init_db()
