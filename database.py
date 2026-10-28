"""
Database module for AI-Powered System Resource Manager.
Provides persistent SQLite storage for timestamped system metrics and detected anomaly events.
"""

import sqlite3
import os
from typing import Dict, Any, Optional
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_monitor.db")


_initialized_paths = set()

def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database, ensuring tables exist."""
    target_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(target_path, timeout=10.0)
    conn.row_factory = sqlite3.Row

    if target_path not in _initialized_paths:
        _init_tables(conn)
        _initialized_paths.add(target_path)

    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    """Internal helper to create schema on an open connection."""
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_percent REAL NOT NULL,
                memory_percent REAL NOT NULL,
                memory_used_gb REAL NOT NULL,
                memory_available_gb REAL NOT NULL,
                disk_percent REAL NOT NULL,
                net_sent_kb REAL NOT NULL,
                net_recv_kb REAL NOT NULL,
                process_count INTEGER NOT NULL,
                anomaly_score REAL DEFAULT 0.0,
                is_anomaly INTEGER DEFAULT 0
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_percent REAL NOT NULL,
                memory_percent REAL NOT NULL,
                disk_percent REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                explanation TEXT NOT NULL,
                top_process TEXT,
                top_process_cpu REAL
            );
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_ts ON system_metrics(timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_ts ON anomaly_events(timestamp);")


def init_db(db_path: Optional[str] = None) -> None:
    """Initializes SQLite tables for metrics and anomaly logs if they do not exist."""
    conn = get_connection(db_path)
    conn.close()


def insert_metric(metric: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """
    Inserts a single timestamped metric snapshot into the system_metrics table.
    Returns the inserted row ID.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("""
                INSERT INTO system_metrics (
                    timestamp, cpu_percent, memory_percent, memory_used_gb,
                    memory_available_gb, disk_percent, net_sent_kb, net_recv_kb,
                    process_count, anomaly_score, is_anomaly
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.get("timestamp"),
                float(metric.get("cpu_percent", 0.0)),
                float(metric.get("memory_percent", 0.0)),
                float(metric.get("memory_used_gb", 0.0)),
                float(metric.get("memory_available_gb", 0.0)),
                float(metric.get("disk_percent", 0.0)),
                float(metric.get("net_sent_kb", 0.0)),
                float(metric.get("net_recv_kb", 0.0)),
                int(metric.get("process_count", 0)),
                float(metric.get("anomaly_score", 0.0)),
                int(1 if metric.get("is_anomaly") else 0)
            ))
            return cursor.lastrowid
    finally:
        conn.close()


def insert_anomaly_event(event: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """
    Inserts a detected anomaly event record.
    Returns the inserted row ID.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("""
                INSERT INTO anomaly_events (
                    timestamp, cpu_percent, memory_percent, disk_percent,
                    anomaly_score, explanation, top_process, top_process_cpu
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.get("timestamp"),
                float(event.get("cpu_percent", 0.0)),
                float(event.get("memory_percent", 0.0)),
                float(event.get("disk_percent", 0.0)),
                float(event.get("anomaly_score", 0.0)),
                str(event.get("explanation", "")),
                str(event.get("top_process", "N/A")),
                float(event.get("top_process_cpu", 0.0))
            ))
            return cursor.lastrowid
    finally:
        conn.close()


def get_recent_metrics(limit: int = 100, db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieves the most recent system metrics ordered chronologically for plotting.
    Returns a pandas DataFrame.
    """
    conn = get_connection(db_path)
    try:
        query = f"""
            SELECT * FROM (
                SELECT * FROM system_metrics
                ORDER BY id DESC
                LIMIT {int(limit)}
            ) ORDER BY id ASC;
        """
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def get_recent_anomalies(limit: int = 50, db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieves the most recent detected anomaly events in descending order (latest first).
    """
    conn = get_connection(db_path)
    try:
        query = f"""
            SELECT * FROM anomaly_events
            ORDER BY id DESC
            LIMIT {int(limit)};
        """
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def get_all_metrics_dataframe(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieves all stored metrics for ML model retraining or historical analysis.
    """
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM system_metrics ORDER BY id ASC;", conn)
        return df
    finally:
        conn.close()


def get_metrics_count(db_path: Optional[str] = None) -> int:
    """Returns total count of recorded metric rows."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM system_metrics;")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def get_anomaly_count(db_path: Optional[str] = None) -> int:
    """Returns total count of detected anomaly events."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM anomaly_events;")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def clear_history(db_path: Optional[str] = None) -> bool:
    """
    Clears all stored metrics and anomaly events.
    Returns True upon successful deletion.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM system_metrics;")
            conn.execute("DELETE FROM anomaly_events;")
        conn.isolation_level = None
        conn.execute("VACUUM;")
        return True
    except Exception:
        return False
    finally:
        conn.close()


def prune_old_metrics(max_records: int = 5000, db_path: Optional[str] = None) -> int:
    """
    Keeps database size bounded by deleting metrics older than the most recent max_records.
    Returns the number of deleted records.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("""
                DELETE FROM system_metrics
                WHERE id NOT IN (
                    SELECT id FROM system_metrics
                    ORDER BY id DESC
                    LIMIT ?
                );
            """, (max_records,))
            return cursor.rowcount
    finally:
        conn.close()
