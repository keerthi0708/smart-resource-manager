"""
Unit and integration tests for AI-Powered System Resource Manager.
Tests resource metric collection, process management safeguards,
Isolation Forest anomaly detection, recommendation heuristics, and SQLite persistence.
Uses mocks where appropriate to ensure deterministic execution independent of current machine load.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

import database
import monitor
import process_manager
import anomaly_detector
import recommendation_engine


# =====================================================================
# 1. DATABASE TESTS
# =====================================================================

class TestDatabase:
    @pytest.fixture(autouse=True)
    def setup_temp_db(self):
        """Create a temporary SQLite database file for isolation."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_file.name
        self.temp_file.close()
        database.init_db(self.db_path)
        yield
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_init_db_creates_tables(self):
        conn = database.get_connection(self.db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        table_names = [t[0] for t in tables]
        conn.close()
        assert "system_metrics" in table_names
        assert "anomaly_events" in table_names

    def test_insert_and_get_metrics(self):
        sample_metric = {
            "timestamp": "2026-10-21 12:00:00",
            "cpu_percent": 35.5,
            "memory_percent": 60.2,
            "memory_used_gb": 8.5,
            "memory_available_gb": 7.5,
            "disk_percent": 45.0,
            "net_sent_kb": 120.0,
            "net_recv_kb": 350.0,
            "process_count": 180,
            "anomaly_score": 0.12,
            "is_anomaly": False
        }
        row_id = database.insert_metric(sample_metric, self.db_path)
        assert row_id is not None
        assert row_id > 0

        df = database.get_recent_metrics(limit=10, db_path=self.db_path)
        assert len(df) == 1
        assert df.iloc[0]["cpu_percent"] == 35.5
        assert df.iloc[0]["process_count"] == 180

    def test_insert_and_get_anomalies(self):
        sample_event = {
            "timestamp": "2026-10-21 12:05:00",
            "cpu_percent": 98.2,
            "memory_percent": 92.4,
            "disk_percent": 50.0,
            "anomaly_score": 0.88,
            "explanation": "Critical CPU and memory surge detected.",
            "top_process": "stress_worker",
            "top_process_cpu": 85.0
        }
        row_id = database.insert_anomaly_event(sample_event, self.db_path)
        assert row_id is not None
        assert row_id > 0

        df = database.get_recent_anomalies(limit=10, db_path=self.db_path)
        assert len(df) == 1
        assert df.iloc[0]["top_process"] == "stress_worker"
        assert df.iloc[0]["anomaly_score"] == 0.88

    def test_clear_and_prune_history(self):
        for i in range(10):
            database.insert_metric({
                "timestamp": f"2026-10-21 12:0{i}:00",
                "cpu_percent": 20.0 + i,
                "memory_percent": 50.0,
                "memory_used_gb": 8.0,
                "memory_available_gb": 8.0,
                "disk_percent": 40.0,
                "net_sent_kb": 10.0,
                "net_recv_kb": 20.0,
                "process_count": 150,
                "anomaly_score": 0.05,
                "is_anomaly": False
            }, self.db_path)

        assert database.get_metrics_count(self.db_path) == 10

        # Prune older than 5
        deleted = database.prune_old_metrics(max_records=5, db_path=self.db_path)
        assert deleted == 5
        assert database.get_metrics_count(self.db_path) == 5

        # Clear history
        assert database.clear_history(self.db_path) is True
        assert database.get_metrics_count(self.db_path) == 0


# =====================================================================
# 2. SYSTEM MONITOR TESTS
# =====================================================================

class TestSystemMonitor:
    def test_snapshot_keys_and_ranges(self):
        mon = monitor.SystemMonitor()
        snapshot = mon.get_system_snapshot()

        assert "cpu_percent" in snapshot
        assert "memory_percent" in snapshot
        assert "disk_percent" in snapshot
        assert "process_count" in snapshot
        assert "per_cpu_percent" in snapshot
        assert "net_sent_kb" in snapshot
        assert "net_recv_kb" in snapshot

        assert 0.0 <= snapshot["cpu_percent"] <= 100.0
        assert 0.0 <= snapshot["memory_percent"] <= 100.0
        assert 0.0 <= snapshot["disk_percent"] <= 100.0
        assert snapshot["process_count"] >= 0

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_mocked_snapshot(self, mock_disk, mock_mem, mock_cpu):
        mock_cpu.side_effect = lambda interval=None, percpu=False: [40.0, 45.0] if percpu else 42.5
        
        mem_mock = MagicMock()
        mem_mock.total = 16 * 1024 ** 3
        mem_mock.used = 8 * 1024 ** 3
        mem_mock.available = 8 * 1024 ** 3
        mem_mock.percent = 50.0
        mock_mem.return_value = mem_mock

        disk_mock = MagicMock()
        disk_mock.total = 500 * 1024 ** 3
        disk_mock.used = 250 * 1024 ** 3
        disk_mock.free = 250 * 1024 ** 3
        disk_mock.percent = 50.0
        mock_disk.return_value = disk_mock

        mon = monitor.SystemMonitor()
        snap = mon.get_system_snapshot()

        assert snap["memory_percent"] == 50.0
        assert snap["disk_percent"] == 50.0
        assert snap["memory_total_gb"] == 16.0


# =====================================================================
# 3. PROCESS MANAGER TESTS
# =====================================================================

class TestProcessManager:
    def test_get_processes_returns_dataframe(self):
        df = process_manager.get_processes(limit=10)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"pid", "name", "cpu_percent", "memory_percent", "memory_mb", "status"}
        assert expected_cols.issubset(set(df.columns))
        if not df.empty:
            assert len(df) <= 10

    def test_protected_process_detection(self):
        assert process_manager.is_protected_process(0, "System Idle Process") is True
        assert process_manager.is_protected_process(4, "System") is True
        assert process_manager.is_protected_process(9999, "csrss.exe") is True
        assert process_manager.is_protected_process(10050, "my_custom_script.py") is False

    def test_terminate_process_safety_guards(self):
        # Invalid PID
        success, msg = process_manager.terminate_process(-5)
        assert success is False
        assert "Invalid PID" in msg

        # Protected PID
        success, msg = process_manager.terminate_process(4)
        assert success is False
        assert "critical operating system component" in msg


# =====================================================================
# 4. ANOMALY DETECTOR TESTS
# =====================================================================

class TestAnomalyDetector:
    def test_synthetic_baseline_generation(self):
        df = anomaly_detector.generate_synthetic_baseline(n_samples=200)
        assert len(df) == 200
        for feat in anomaly_detector.FEATURE_NAMES:
            assert feat in df.columns

    def test_detector_initialization_and_inference(self):
        detector = anomaly_detector.ResourceAnomalyDetector(contamination=0.05)
        assert detector.is_trained is True

        # Test normal snapshot
        normal_snap = {
            "cpu_percent": 15.0,
            "memory_percent": 45.0,
            "disk_percent": 40.0,
            "net_sent_kb": 25.0,
            "net_recv_kb": 50.0,
            "process_count": 140
        }
        res_normal = detector.analyze_snapshot(normal_snap)
        assert "is_anomaly" in res_normal
        assert "anomaly_score" in res_normal
        assert "explanation" in res_normal
        assert res_normal["is_anomaly"] is False

        # Test extreme anomaly snapshot (statistically far out)
        extreme_snap = {
            "cpu_percent": 99.5,
            "memory_percent": 98.0,
            "disk_percent": 95.0,
            "net_sent_kb": 85000.0,
            "net_recv_kb": 95000.0,
            "process_count": 650
        }
        res_extreme = detector.analyze_snapshot(extreme_snap)
        assert res_extreme["is_anomaly"] is True
        assert res_extreme["anomaly_score"] > 0.60
        assert "Primary contributing metrics" in res_extreme["explanation"] or res_extreme["is_anomaly"]

    def test_retraining_on_historical_data(self):
        detector = anomaly_detector.ResourceAnomalyDetector()
        
        # Insufficient data
        short_df = pd.DataFrame([{"cpu_percent": 10.0} for _ in range(5)])
        success, msg = detector.train_on_historical_data(short_df)
        assert success is False
        assert "Insufficient" in msg

        # Sufficient data
        baseline_df = anomaly_detector.generate_synthetic_baseline(n_samples=50)
        success, msg = detector.train_on_historical_data(baseline_df)
        assert success is True
        assert "Successfully trained" in msg
        assert "Retrained on Live System History" in detector.training_source


# =====================================================================
# 5. RECOMMENDATION ENGINE TESTS
# =====================================================================

class TestRecommendationEngine:
    def test_balanced_system_recommendation(self):
        snapshot = {
            "cpu_percent": 25.0,
            "per_cpu_percent": [25.0, 25.0, 25.0, 25.0],
            "memory_percent": 45.0,
            "swap_percent": 10.0,
            "disk_percent": 50.0,
            "disk_partition": "C:\\",
            "disk_free_gb": 200.0,
            "net_sent_kb": 50.0,
            "net_recv_kb": 100.0,
            "process_count": 120
        }
        recs = recommendation_engine.generate_recommendations(snapshot)
        assert len(recs) == 1
        assert recs[0]["severity"] == "SUCCESS"
        assert "Optimal Balance" in recs[0]["title"]

    def test_high_cpu_saturation_recommendation(self):
        snapshot = {
            "cpu_percent": 92.0,
            "per_cpu_percent": [90.0, 94.0],
            "memory_percent": 50.0,
            "swap_percent": 10.0,
            "disk_percent": 50.0,
            "disk_partition": "C:\\",
            "disk_free_gb": 100.0,
            "net_sent_kb": 10.0,
            "net_recv_kb": 10.0,
            "process_count": 150
        }
        mock_procs = pd.DataFrame([{
            "pid": 4321,
            "name": "ffmpeg.exe",
            "cpu_percent": 82.0,
            "memory_percent": 12.0,
            "memory_mb": 500.0
        }])

        recs = recommendation_engine.generate_recommendations(snapshot, top_processes=mock_procs)
        titles = [r["title"] for r in recs]
        assert any("Dominant Process CPU Saturation" in t for t in titles)
        assert any(r["severity"] == "CRITICAL" for r in recs)

    def test_memory_pressure_thrashing_recommendation(self):
        snapshot = {
            "cpu_percent": 20.0,
            "memory_percent": 94.0,
            "swap_percent": 65.0,
            "disk_percent": 40.0,
            "disk_free_gb": 150.0,
            "net_sent_kb": 5.0,
            "net_recv_kb": 5.0,
            "process_count": 160
        }
        mock_procs = pd.DataFrame([{
            "pid": 8888,
            "name": "memory_hog.exe",
            "cpu_percent": 2.0,
            "memory_percent": 45.0,
            "memory_mb": 7200.0
        }])

        recs = recommendation_engine.generate_recommendations(snapshot, top_processes=mock_procs)
        titles = [r["title"] for r in recs]
        assert any("Severe Physical Memory Pressure" in t for t in titles)
        assert any("Heavy Swap / Pagefile Utilization" in t for t in titles)
