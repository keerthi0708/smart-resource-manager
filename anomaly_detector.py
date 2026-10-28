"""
Machine Learning Anomaly Detection module for AI-Powered System Resource Manager.
Implements an Isolation Forest model using Scikit-Learn to detect multidimensional
resource consumption anomalies in real time with feature attribution and explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "net_sent_kb",
    "net_recv_kb",
    "process_count"
]

FEATURE_LABELS = {
    "cpu_percent": "CPU Utilization (%)",
    "memory_percent": "Memory Utilization (%)",
    "disk_percent": "Disk Usage (%)",
    "net_sent_kb": "Network Upload (KB/s)",
    "net_recv_kb": "Network Download (KB/s)",
    "process_count": "Active Process Count"
}


def generate_synthetic_baseline(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic multi-modal desktop baseline dataset for model bootstrapping.
    Clearly designated as synthetic data representing common PC workloads:
    - Idle mode (~40% of time)
    - Office/Browsing mode (~45% of time)
    - Heavy compilation/multimedia mode (~15% of time)
    """
    rng = np.random.default_rng(random_state)
    n_idle = int(n_samples * 0.40)
    n_work = int(n_samples * 0.45)
    n_heavy = n_samples - n_idle - n_work

    # 1. Idle state
    idle_cpu = rng.uniform(2.0, 15.0, n_idle)
    idle_mem = rng.uniform(30.0, 50.0, n_idle)
    idle_disk = rng.uniform(35.0, 55.0, n_idle)
    idle_sent = rng.exponential(15.0, n_idle)
    idle_recv = rng.exponential(35.0, n_idle)
    idle_procs = rng.integers(110, 170, n_idle)

    # 2. Standard Productivity state
    work_cpu = rng.uniform(15.0, 45.0, n_work)
    work_mem = rng.uniform(45.0, 68.0, n_work)
    work_disk = rng.uniform(40.0, 60.0, n_work)
    work_sent = rng.exponential(80.0, n_work)
    work_recv = rng.exponential(250.0, n_work)
    work_procs = rng.integers(150, 230, n_work)

    # 3. Heavy Compute state
    heavy_cpu = rng.uniform(45.0, 80.0, n_heavy)
    heavy_mem = rng.uniform(65.0, 85.0, n_heavy)
    heavy_disk = rng.uniform(45.0, 75.0, n_heavy)
    heavy_sent = rng.exponential(300.0, n_heavy)
    heavy_recv = rng.exponential(800.0, n_heavy)
    heavy_procs = rng.integers(180, 260, n_heavy)

    data = {
        "cpu_percent": np.clip(np.concatenate([idle_cpu, work_cpu, heavy_cpu]), 0.0, 100.0),
        "memory_percent": np.clip(np.concatenate([idle_mem, work_mem, heavy_mem]), 0.0, 100.0),
        "disk_percent": np.clip(np.concatenate([idle_disk, work_disk, heavy_disk]), 0.0, 100.0),
        "net_sent_kb": np.clip(np.concatenate([idle_sent, work_sent, heavy_sent]), 0.0, 50000.0),
        "net_recv_kb": np.clip(np.concatenate([idle_recv, work_recv, heavy_recv]), 0.0, 50000.0),
        "process_count": np.concatenate([idle_procs, work_procs, heavy_procs])
    }

    return pd.DataFrame(data)


class ResourceAnomalyDetector:
    """
    Isolation Forest anomaly detector that identifies statistical deviations in
    multivariate system resource consumption.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self.training_source = "Uninitialized"
        self.training_samples_count = 0
        self.feature_stats: Dict[str, Dict[str, float]] = {}

        # Initialize with synthetic baseline so model is immediately functional
        self.train_on_synthetic_baseline()

    def train_on_synthetic_baseline(self, n_samples: int = 600) -> None:
        """Trains the detector on a realistic synthetic desktop baseline."""
        df_synthetic = generate_synthetic_baseline(n_samples=n_samples, random_state=self.random_state)
        self._fit_dataframe(df_synthetic, source_label=f"Pre-trained Baseline ({n_samples} synthetic samples)")

    def train_on_historical_data(self, history_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Retrains the model using actual collected system readings from the SQLite database.
        Requires at least 25 historical records to ensure valid statistical diversity.
        """
        if history_df is None or len(history_df) < 25:
            count = len(history_df) if history_df is not None else 0
            return False, f"Insufficient historical data: {count} samples found (minimum 25 required for retraining)."

        # Ensure all required features are present
        missing = [f for f in FEATURE_NAMES if f not in history_df.columns]
        if missing:
            return False, f"Missing required feature columns in history: {missing}"

        # Clean valid numeric data
        clean_df = history_df[FEATURE_NAMES].dropna()
        if len(clean_df) < 25:
            return False, "Insufficient non-null rows in history."

        self._fit_dataframe(clean_df, source_label=f"Retrained on Live System History ({len(clean_df)} real observations)")
        return True, f"Successfully trained model on {len(clean_df)} real historical observations."

    def _fit_dataframe(self, df: pd.DataFrame, source_label: str) -> None:
        """Fits standard scaler and Isolation Forest on the provided dataset."""
        X = df[FEATURE_NAMES].values
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled)

        # Store baseline distribution statistics for feature attribution
        self.feature_stats = {}
        for feat in FEATURE_NAMES:
            self.feature_stats[feat] = {
                "mean": float(df[feat].mean()),
                "std": float(max(df[feat].std(), 1e-3)),
                "p95": float(np.percentile(df[feat], 95)),
                "min": float(df[feat].min()),
                "max": float(df[feat].max()),
            }

        self.is_trained = True
        self.training_source = source_label
        self.training_samples_count = len(df)

    def analyze_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs real ML inference on a point-in-time system snapshot.
        Calculates anomaly score, flags abnormal patterns, and identifies contributing metrics.
        """
        if not self.is_trained:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "severity": "NORMAL",
                "explanation": "Model warming up. Initializing baseline distributions.",
                "feature_deviations": {},
                "top_contributor": "None",
                "training_source": self.training_source
            }

        # Extract features vector
        sample_values = [float(snapshot.get(feat, 0.0)) for feat in FEATURE_NAMES]
        X = np.array([sample_values])
        X_scaled = self.scaler.transform(X)

        # Isolation forest inference: 1 = normal, -1 = anomaly
        pred = self.model.predict(X_scaled)[0]
        # Raw decision function: positive = normal, negative = anomaly
        raw_score = float(self.model.decision_function(X_scaled)[0])

        # Normalize score into intuitive scale: 0.0 (very normal) to 1.0 (severe anomaly)
        # Decision function is typically in range [-0.3, +0.3]
        normalized_anomaly_prob = float(np.clip(0.5 - (raw_score * 2.0), 0.0, 1.0))
        is_anomaly = bool(pred == -1 or normalized_anomaly_prob > 0.65)

        # Feature Attribution: measure how many std deviations each metric departs from baseline
        deviations: Dict[str, Dict[str, Any]] = {}
        highest_z = 0.0
        top_contributor = "None"

        for idx, feat in enumerate(FEATURE_NAMES):
            val = sample_values[idx]
            stats = self.feature_stats.get(feat, {"mean": 0.0, "std": 1.0})
            mean = stats["mean"]
            std = stats["std"]
            z_score = abs(val - mean) / std

            deviations[feat] = {
                "label": FEATURE_LABELS[feat],
                "current_value": round(val, 2),
                "baseline_mean": round(mean, 2),
                "baseline_std": round(std, 2),
                "z_score": round(z_score, 2),
                "is_deviated": bool(z_score >= 2.0)
            }

            if z_score > highest_z:
                highest_z = z_score
                top_contributor = FEATURE_LABELS[feat]

        # Construct contextual explanation
        explanation = self._build_explanation(is_anomaly, deviations, normalized_anomaly_prob, snapshot)

        severity = "NORMAL"
        if is_anomaly:
            severity = "HIGH ANOMALY" if normalized_anomaly_prob >= 0.80 else "MODERATE ANOMALY"

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(normalized_anomaly_prob, 3),
            "raw_decision_score": round(raw_score, 4),
            "severity": severity,
            "explanation": explanation,
            "feature_deviations": deviations,
            "top_contributor": top_contributor,
            "training_source": self.training_source,
            "training_samples_count": self.training_samples_count
        }

    def _build_explanation(
        self,
        is_anomaly: bool,
        deviations: Dict[str, Dict[str, Any]],
        anomaly_score: float,
        snapshot: Dict[str, Any]
    ) -> str:
        """
        Builds a human-readable, technically accurate explanation of why an observation
        was flagged or deemed normal, without alarming false claims of malware.
        """
        deviated_feats = [d for d in deviations.values() if d["is_deviated"]]

        if not is_anomaly and not deviated_feats:
            return (
                "System resource patterns match established operational baselines. "
                "Multivariate CPU, RAM, disk, and network combinations are within expected limits."
            )

        if not is_anomaly and deviated_feats:
            feat_names = ", ".join([d["label"] for d in deviated_feats])
            return (
                f"Elevated reading noted in {feat_names}, but overall multidimensional interaction "
                "remains within acceptable cluster boundaries. No severe anomaly triggered."
            )

        # It is an anomaly
        reasons = []
        for d in sorted(deviated_feats, key=lambda x: x["z_score"], reverse=True):
            reasons.append(
                f"{d['label']} at {d['current_value']} (baseline avg: {d['baseline_mean']}, "
                f"+{d['z_score']}σ deviation)"
            )

        if not reasons:
            explanation = (
                f"Statistical anomaly detected (Anomaly Index: {int(anomaly_score * 100)}%). "
                "An unusual concurrent combination of system resources was isolated by the tree ensemble, "
                "even though individual single-metric thresholds were not drastically breached."
            )
        else:
            explanation = (
                f"Statistical anomaly detected (Anomaly Index: {int(anomaly_score * 100)}%). "
                f"Primary contributing metrics: {'; '.join(reasons)}. "
                "Note: A statistical anomaly denotes an atypical resource footprint (e.g. heavy compilation, "
                "bulk file transfer, or batch compression) and is not inherently indicative of a software fault."
            )

        return explanation


# Singleton detector instance for consistent state in Streamlit session
_global_detector: Optional[ResourceAnomalyDetector] = None


def get_anomaly_detector() -> ResourceAnomalyDetector:
    """Returns or creates the shared ResourceAnomalyDetector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = ResourceAnomalyDetector()
    return _global_detector
