"""
AI-Powered System Resource Manager
Main Streamlit Dashboard Application.
Integrates live psutil monitoring, process management with confirmed termination,
Scikit-Learn Isolation Forest anomaly detection, deterministic OS recommendation heuristics,
and SQLite historical tracking.
"""

import time
import datetime
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import database
import monitor
import process_manager
import anomaly_detector
import recommendation_engine

# =====================================================================
# STREAMLIT PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="AI-Powered System Resource Manager",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, readable UI
st.markdown("""
<style>
    .metric-card {
        background: #1e222d;
        border: 1px solid #2e3646;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.8rem;
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .badge-warning {
        background-color: #f59e0b;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .badge-info {
        background-color: #3b82f6;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .badge-success {
        background-color: #10b981;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .rule-tag {
        font-size: 0.7rem;
        border: 1px solid #475569;
        border-radius: 3px;
        padding: 2px 6px;
        color: #94a3b8;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# SESSION STATE & SYSTEM INITIALIZATION
# =====================================================================
@st.cache_resource
def initialize_system():
    """Initializes SQLite database and singleton services."""
    database.init_db()
    mon = monitor.get_monitor()
    detector = anomaly_detector.get_anomaly_detector()
    return mon, detector


sys_monitor, ml_detector = initialize_system()

if "confirm_terminate_pid" not in st.session_state:
    st.session_state.confirm_terminate_pid = None
if "last_action_msg" not in st.session_state:
    st.session_state.last_action_msg = None
if "confirm_clear_db" not in st.session_state:
    st.session_state.confirm_clear_db = False


# =====================================================================
# SIDEBAR CONTROLS & MONITORING TELEMETRY
# =====================================================================
st.sidebar.title("⚡ Resource Manager")
st.sidebar.caption("Undergraduate Operating Systems & ML Project")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Monitor Settings")

auto_refresh = st.sidebar.toggle("Auto-Refresh Dashboard", value=False)
refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)",
    min_value=2,
    max_value=15,
    value=4,
    step=1,
    disabled=not auto_refresh
)

manual_refresh = st.sidebar.button("🔄 Refresh Data Now", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🖥️ Machine Specifications")
st.sidebar.markdown(f"**Host OS:** `{os.name.upper()}`")
st.sidebar.markdown(f"**Logical Cores:** `{sys_monitor.logical_cores}`")
st.sidebar.markdown(f"**Physical Cores:** `{sys_monitor.physical_cores}`")
snapshot_quick = sys_monitor.get_system_snapshot()
st.sidebar.markdown(f"**System Uptime:** `{snapshot_quick.get('uptime_str', 'N/A')}`")
st.sidebar.markdown(f"**Root Volume:** `{snapshot_quick.get('disk_partition', 'N/A')}`")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Model Status")
st.sidebar.info(f"**Model:** Isolation Forest\n\n**Source:**\n{ml_detector.training_source}")

# Retraining trigger
if st.sidebar.button("⚡ Retrain Model on Live History", use_container_width=True):
    history_all = database.get_all_metrics_dataframe()
    success, retrain_msg = ml_detector.train_on_historical_data(history_all)
    if success:
        st.sidebar.success(retrain_msg)
    else:
        st.sidebar.warning(retrain_msg)


# =====================================================================
# LIVE DATA COLLECTION & INFERENCE
# =====================================================================
snapshot = sys_monitor.get_system_snapshot()
procs_df = process_manager.get_processes(sort_by="cpu_percent", ascending=False, limit=100)

# Run genuine ML inference
ml_result = ml_detector.analyze_snapshot(snapshot)

# Store observation in SQLite database
database.insert_metric({
    "timestamp": snapshot["timestamp"],
    "cpu_percent": snapshot["cpu_percent"],
    "memory_percent": snapshot["memory_percent"],
    "memory_used_gb": snapshot["memory_used_gb"],
    "memory_available_gb": snapshot["memory_available_gb"],
    "disk_percent": snapshot["disk_percent"],
    "net_sent_kb": snapshot["net_sent_kb"],
    "net_recv_kb": snapshot["net_recv_kb"],
    "process_count": snapshot["process_count"],
    "anomaly_score": ml_result["anomaly_score"],
    "is_anomaly": ml_result["is_anomaly"]
})

# If an anomaly was detected, log event in database
if ml_result["is_anomaly"]:
    top_proc_name = "N/A"
    top_proc_cpu = 0.0
    if not procs_df.empty:
        top_proc_name = procs_df.iloc[0]["name"]
        top_proc_cpu = procs_df.iloc[0]["cpu_percent"]

    database.insert_anomaly_event({
        "timestamp": snapshot["timestamp"],
        "cpu_percent": snapshot["cpu_percent"],
        "memory_percent": snapshot["memory_percent"],
        "disk_percent": snapshot["disk_percent"],
        "anomaly_score": ml_result["anomaly_score"],
        "explanation": ml_result["explanation"],
        "top_process": top_proc_name,
        "top_process_cpu": top_proc_cpu
    })

# Generate deterministic rule-based recommendations
recommendations = recommendation_engine.generate_recommendations(
    snapshot=snapshot,
    top_processes=procs_df,
    ml_result=ml_result
)


# =====================================================================
# DASHBOARD TABS
# =====================================================================
tab_overview, tab_procs, tab_ai, tab_recs, tab_history, tab_concepts = st.tabs([
    "📊 Overview",
    "⚙️ Process Manager",
    "🧠 AI Anomaly Analysis",
    "💡 Recommendations",
    "📈 Resource History",
    "📖 OS Concepts & Architecture"
])


# =====================================================================
# TAB 1: OVERVIEW
# =====================================================================
with tab_overview:
    st.subheader("🖥️ Real-Time System Metrics")

    # Metric Cards Row
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric(
            label="CPU Utilization",
            value=f"{snapshot['cpu_percent']}%",
            delta=f"{round(snapshot['cpu_freq_current'] / 1000, 2)} GHz" if snapshot['cpu_freq_current'] else "Active"
        )
    with m2:
        st.metric(
            label="RAM Utilization",
            value=f"{snapshot['memory_percent']}%",
            delta=f"{snapshot['memory_used_gb']} / {snapshot['memory_total_gb']} GB"
        )
    with m3:
        st.metric(
            label="System Disk",
            value=f"{snapshot['disk_percent']}%",
            delta=f"{snapshot['disk_free_gb']} GB free"
        )
    with m4:
        net_rate_total = snapshot['net_sent_kb'] + snapshot['net_recv_kb']
        st.metric(
            label="Network Throughput",
            value=f"{round(net_rate_total, 1)} KB/s",
            delta=f"↓{round(snapshot['net_recv_kb'], 1)} | ↑{round(snapshot['net_sent_kb'], 1)}"
        )
    with m5:
        st.metric(
            label="Active Processes",
            value=snapshot['process_count'],
            delta=f"Context switches: {snapshot['context_switches']:,}" if snapshot['context_switches'] else "Stable"
        )

    st.markdown("---")

    # Time-series trend chart & Per-Core CPU Scheduling Breakdown
    c_left, c_right = st.columns([1.5, 1])

    with c_left:
        st.markdown("##### 📈 Live Resource Utilization Trends")
        recent_df = database.get_recent_metrics(limit=60)
        if not recent_df.empty:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=recent_df["timestamp"],
                y=recent_df["cpu_percent"],
                mode="lines",
                name="CPU (%)",
                line=dict(color="#3b82f6", width=2.5)
            ))
            fig_trend.add_trace(go.Scatter(
                x=recent_df["timestamp"],
                y=recent_df["memory_percent"],
                mode="lines",
                name="RAM (%)",
                line=dict(color="#10b981", width=2.5)
            ))
            fig_trend.add_trace(go.Scatter(
                x=recent_df["timestamp"],
                y=recent_df["disk_percent"],
                mode="lines",
                name="Disk (%)",
                line=dict(color="#f59e0b", width=1.5, dash="dot")
            ))
            fig_trend.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(range=[0, 105], title="Utilization (%)"),
                xaxis=dict(showticklabels=False)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Gathering initial telemetry readings...")

    with c_right:
        st.markdown("##### 🔀 CPU Multi-Core Scheduling Distribution")
        per_cpu = snapshot.get("per_cpu_percent", [])
        if per_cpu:
            core_df = pd.DataFrame({
                "Core": [f"Core {i}" for i in range(len(per_cpu))],
                "Load": per_cpu
            })
            fig_cores = px.bar(
                core_df,
                x="Core",
                y="Load",
                text="Load",
                labels={"Load": "Usage (%)"},
                color="Load",
                color_continuous_scale="Blues",
                range_y=[0, 100]
            )
            fig_cores.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_cores.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=20, b=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_cores, use_container_width=True)
        else:
            st.info("Per-core CPU metrics not accessible.")

    # Second row: Memory breakdown & I/O
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("##### 💾 Virtual Memory & Swap Paging")
        mem_labels = ["RAM Used", "RAM Available", "Swap Used"]
        mem_values = [
            snapshot["memory_used_gb"],
            snapshot["memory_available_gb"],
            snapshot["swap_used_gb"]
        ]
        fig_mem = go.Figure(data=[go.Pie(
            labels=mem_labels,
            values=mem_values,
            hole=.4,
            marker_colors=["#ef4444", "#10b981", "#8b5cf6"]
        )])
        fig_mem.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_mem, use_container_width=True)

    with row2_col2:
        st.markdown("##### 🌐 I/O Activity Rates")
        io_df = pd.DataFrame({
            "I/O Subsystem": ["Disk Read", "Disk Write", "Net Recv (Down)", "Net Sent (Up)"],
            "Rate (KB/s)": [
                snapshot.get("disk_read_kbps", 0.0),
                snapshot.get("disk_write_kbps", 0.0),
                snapshot.get("net_recv_kb", 0.0),
                snapshot.get("net_sent_kb", 0.0)
            ]
        })
        fig_io = px.bar(
            io_df,
            x="I/O Subsystem",
            y="Rate (KB/s)",
            color="I/O Subsystem",
            text="Rate (KB/s)"
        )
        fig_io.update_traces(texttemplate="%{text:.1f} KB/s", textposition="outside")
        fig_io.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_io, use_container_width=True)


# =====================================================================
# TAB 2: PROCESS MANAGER
# =====================================================================
with tab_procs:
    st.subheader("⚙️ Process Control & Inspection")
    st.caption("Demonstrating Process Control Block (PCB) attributes, scheduling status, and memory residency.")

    # Action result message banner
    if st.session_state.last_action_msg:
        success, msg = st.session_state.last_action_msg
        if success:
            st.success(msg)
        else:
            st.error(msg)
        st.session_state.last_action_msg = None

    # Search & Sorting Controls
    c_search, c_sort, c_order, c_limit = st.columns([2, 1.5, 1, 1])

    with c_search:
        search_query = st.text_input("🔍 Filter by Process Name or PID", placeholder="e.g. python, chrome, 4321")
    with c_sort:
        sort_col = st.selectbox(
            "Sort By",
            options=["cpu_percent", "memory_percent", "memory_mb", "num_threads", "pid", "name"],
            format_func=lambda x: {
                "cpu_percent": "CPU Usage (%)",
                "memory_percent": "Memory (%)",
                "memory_mb": "Resident RAM (MB)",
                "num_threads": "Thread Count",
                "pid": "Process ID (PID)",
                "name": "Process Name"
            }.get(x, x)
        )
    with c_order:
        ascending = st.selectbox("Order", options=[False, True], format_func=lambda x: "Descending" if not x else "Ascending")
    with c_limit:
        proc_limit = st.selectbox("Limit", options=[25, 50, 100, 200], index=1)

    filtered_procs = process_manager.get_processes(
        sort_by=sort_col,
        ascending=ascending,
        limit=proc_limit,
        search=search_query
    )

    # Format table for visual presentation
    display_df = filtered_procs.copy()
    display_df.rename(columns={
        "pid": "PID",
        "name": "Process Name",
        "cpu_percent": "CPU (%)",
        "memory_percent": "Memory (%)",
        "memory_mb": "RAM (MB)",
        "num_threads": "Threads",
        "status": "State",
        "username": "User",
        "created_time": "Started"
    }, inplace=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=380
    )

    st.markdown("---")

    # Safe Process Termination Section
    st.subheader("🛑 Controlled Process Termination")
    st.markdown(
        "Safely signal or terminate a process with **explicit user confirmation**. "
        "Critical operating system processes (e.g. system idle, wininit, csrss, init) are protected."
    )

    p_col1, p_col2 = st.columns([1, 2])

    with p_col1:
        target_pid = st.number_input("Enter PID to Inspect/Terminate", min_value=1, step=1, value=None)
        inspect_btn = st.button("Inspect Process Details", use_container_width=True)

    with p_col2:
        if target_pid:
            details = process_manager.get_process_details(int(target_pid))
            if details:
                st.markdown(f"**Target Process:** `{details['name']}` (PID: `{details['pid']}`)")
                
                # Check protected status
                if details["is_protected"]:
                    st.error("🛡️ **PROTECTED SYSTEM PROCESS:** This process is critical to OS stability and cannot be terminated.")
                else:
                    st.write(
                        f"• **Status:** `{details['status']}` | **Threads:** `{details['num_threads']}`\n"
                        f"• **Current CPU:** `{details['cpu_percent']}%` | **Memory:** `{details['memory_mb']} MB` ({details['memory_percent']}%)\n"
                        f"• **User Account:** `{details['username']}`"
                    )

                    confirm_check = st.checkbox(
                        f"⚠️ I explicitly confirm termination of '{details['name']}' (PID {details['pid']})",
                        key="confirm_term_checkbox"
                    )

                    if st.button("Terminate Process", type="primary", disabled=not confirm_check):
                        success, status_msg = process_manager.terminate_process(int(target_pid))
                        st.session_state.last_action_msg = (success, status_msg)
                        st.rerun()
            else:
                st.warning(f"Process PID {target_pid} could not be inspected or does not exist.")


# =====================================================================
# TAB 3: AI ANOMALY ANALYSIS
# =====================================================================
with tab_ai:
    st.subheader("🧠 Machine Learning Anomaly Detection (Isolation Forest)")
    st.caption(
        "Demonstrates real-time multivariate anomaly detection using an ensemble of isolation trees. "
        "Evaluates the joint distribution of CPU, RAM, Disk, Network, and Process Count."
    )

    # Status Banner
    is_anomaly = ml_result["is_anomaly"]
    score = ml_result["anomaly_score"]
    severity = ml_result["severity"]

    if is_anomaly:
        st.error(f"🚨 **ANOMALY DETECTED** — Severity: **{severity}** (Anomaly Score: {int(score * 100)}%)")
    else:
        st.success(f"✅ **NORMAL OPERATIONAL PROFILE** — System behaves within standard distribution (Score: {int(score * 100)}%)")

    # Metrics and Score Gauge
    a_col1, a_col2 = st.columns([1, 2])

    with a_col1:
        st.markdown("##### 🎯 Anomaly Confidence Meter")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Anomaly Index (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ef4444" if is_anomaly else "#10b981"},
                'steps': [
                    {'range': [0, 50], 'color': "#1e293b"},
                    {'range': [50, 70], 'color': "#334155"},
                    {'range': [70, 100], 'color': "#475569"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 65
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with a_col2:
        st.markdown("##### 📝 Technical Diagnostic Explanation")
        st.info(ml_result["explanation"])

        st.markdown(f"**Model Training Provenance:** `{ml_result['training_source']}`")
        st.markdown(f"**Raw Decision Function Output:** `{ml_result['raw_decision_score']}` *(negative indicates tree isolation)*")

    st.markdown("---")

    # Feature Contribution / Z-score breakdown
    st.markdown("##### 📊 Feature Deviation Analysis (Z-Score Departure from Baseline)")
    st.caption("Measures how many standard deviations ($σ$) current metrics deviate from the established operating mean.")

    deviations = ml_result["feature_deviations"]
    if deviations:
        dev_rows = []
        for feat_name, d in deviations.items():
            dev_rows.append({
                "Metric": d["label"],
                "Current Reading": d["current_value"],
                "Baseline Mean": d["baseline_mean"],
                "Baseline Std (σ)": d["baseline_std"],
                "Z-Score (Deviations)": d["z_score"],
                "Status": "⚠️ Outlier (> 2.0σ)" if d["is_deviated"] else "✅ Nominal"
            })

        dev_df = pd.DataFrame(dev_rows)

        # Plotly horizontal bar chart of Z-Scores
        fig_z = px.bar(
            dev_df,
            x="Z-Score (Deviations)",
            y="Metric",
            orientation="h",
            color="Z-Score (Deviations)",
            color_continuous_scale="Reds",
            text="Z-Score (Deviations)"
        )
        fig_z.add_vline(x=2.0, line_dash="dash", line_color="orange", annotation_text="Outlier Threshold (2.0σ)")
        fig_z.update_traces(texttemplate="%{text:.2f}σ", textposition="outside")
        fig_z.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_z, use_container_width=True)

        st.table(dev_df)


# =====================================================================
# TAB 4: INTELLIGENT RECOMMENDATIONS
# =====================================================================
with tab_recs:
    st.subheader("💡 Intelligent Heuristic Recommendations")
    st.markdown(
        "Rule-based diagnostic feedback analyzing CPU scheduling contention, memory paging, "
        "storage limits, and process table bloat. *Distinct from ML statistical anomaly predictions.*"
    )

    for rec in recommendations:
        sev = rec["severity"]
        cat = rec["category"]
        title = rec["title"]
        desc = rec["description"]
        action = rec["action"]
        rule_tag = rec["rule_type"]

        badge_class = {
            "CRITICAL": "badge-critical",
            "WARNING": "badge-warning",
            "INFO": "badge-info",
            "SUCCESS": "badge-success"
        }.get(sev, "badge-info")

        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <span class="{badge_class}">{sev}</span>
                    <strong style="margin-left: 10px; font-size: 1.05rem;">{title}</strong>
                </div>
                <div>
                    <span class="rule-tag">{cat}</span>
                    <span class="rule-tag">{rule_tag}</span>
                </div>
            </div>
            <p style="color: #cbd5e1; margin-bottom: 6px; font-size: 0.95rem;">{desc}</p>
            <div style="background-color: rgba(255,255,255,0.04); padding: 8px 12px; border-radius: 6px; border-left: 3px solid #3b82f6;">
                <strong style="color: #60a5fa;">Recommended Action:</strong> <span style="color: #e2e8f0;">{action}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# TAB 5: RESOURCE HISTORY
# =====================================================================
with tab_history:
    st.subheader("📈 Historical Resource Logging & Telemetry")
    st.caption("Persistent SQLite storage tracking resource progression and recorded anomaly events.")

    h_col1, h_col2 = st.columns([3, 1])

    with h_col1:
        hist_limit = st.selectbox("Show Last Records", options=[50, 100, 300, 500, 1000], index=1)
    with h_col2:
        total_rows = database.get_metrics_count()
        total_anomalies = database.get_anomaly_count()
        st.metric(label="Total Stored Snapshots", value=total_rows, delta=f"{total_anomalies} Anomalies Logged")

    history_records = database.get_recent_metrics(limit=hist_limit)

    if not history_records.empty:
        # Dual-axis chart: CPU & Memory vs Anomaly Score
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=history_records["timestamp"],
            y=history_records["cpu_percent"],
            name="CPU (%)",
            line=dict(color="#3b82f6", width=2)
        ))
        fig_hist.add_trace(go.Scatter(
            x=history_records["timestamp"],
            y=history_records["memory_percent"],
            name="Memory (%)",
            line=dict(color="#10b981", width=2)
        ))
        fig_hist.add_trace(go.Scatter(
            x=history_records["timestamp"],
            y=history_records["anomaly_score"] * 100,
            name="Anomaly Index (%)",
            line=dict(color="#ef4444", width=1.5, dash="dot")
        ))
        fig_hist.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(range=[0, 105], title="Percentage (%)"),
            xaxis=dict(title="Timestamp", showticklabels=True)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Tabular View
        st.markdown("##### 📋 Measured Telemetry Log")
        st.dataframe(
            history_records[[
                "id", "timestamp", "cpu_percent", "memory_percent",
                "memory_used_gb", "disk_percent", "net_sent_kb",
                "net_recv_kb", "process_count", "anomaly_score", "is_anomaly"
            ]],
            use_container_width=True,
            hide_index=True,
            height=260
        )

        # CSV Download
        csv_data = history_records.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export History to CSV",
            data=csv_data,
            file_name=f"system_resource_history_{datetime.date.today()}.csv",
            mime="text/csv"
        )
    else:
        st.info("No historical records recorded yet.")

    st.markdown("---")

    # Anomaly Events Log
    st.subheader("🚨 Detected Anomaly Log")
    anomalies_df = database.get_recent_anomalies(limit=50)
    if not anomalies_df.empty:
        st.dataframe(anomalies_df, use_container_width=True, hide_index=True)
    else:
        st.success("No anomaly events recorded in SQLite log.")

    # Maintenance Section
    st.markdown("---")
    st.subheader("🧹 Database Maintenance")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        if st.button("Prune Old Metrics (Retain Latest 500)"):
            deleted = database.prune_old_metrics(max_records=500)
            st.success(f"Pruned {deleted} older telemetry records to keep database bounded.")
            st.rerun()

    with d_col2:
        confirm_clear = st.checkbox("Confirm Complete Database History Clear", key="chk_clear_db")
        if st.button("🗑️ Clear All Stored History", type="primary", disabled=not confirm_clear):
            if database.clear_history():
                st.success("Database history wiped successfully.")
                st.rerun()
            else:
                st.error("Failed to clear database.")


# =====================================================================
# TAB 6: OS CONCEPTS & ARCHITECTURE
# =====================================================================
with tab_concepts:
    st.subheader("📖 Operating System Concepts & Architectural Design")

    st.markdown("""
    ### 1. Operating System Concepts Demonstrated

    #### A. CPU Scheduling & Ready Queue Contention
    - **Concept:** The OS scheduler (e.g., Windows Multilevel Feedback Queue, Linux CFS) distributes CPU time slices (quanta) among active threads.
    - **Demonstration:** The dashboard monitors total and per-core CPU utilization. When a single process monopolizes 80%+ of CPU time, it causes thread starvation and latency for interactive tasks. When CPU load is distributed across hundreds of processes, context-switching overhead reduces throughput.

    #### B. Process Control Blocks (PCB) & Process Lifecycle
    - **Concept:** The kernel maintains a PCB for each running process, tracking process state (Running, Sleeping, Stopped, Zombie), PID, parent PID, thread count, and memory allocation.
    - **Demonstration:** The **Process Manager** module inspects live PCB data via non-privileged system calls, exposing resident set size (RSS), thread counts, and process states with safe, confirmed termination guards.

    #### C. Virtual Memory Management & Page Thrashing
    - **Concept:** Modern OSs utilize two-level address translation via page tables, dividing memory into physical frames and virtual pages. When physical memory is saturated, least-recently-used pages are evicted to secondary storage (swap/pagefile).
    - **Demonstration:** The dashboard tracks RAM utilization, available physical frames, and swap/pagefile allocations. If memory exceeds 90% and swap increases, the recommendation engine warns of imminent **page thrashing**.

    #### D. I/O Subsystems & Network Stack
    - **Concept:** Secondary storage (disk) and network network interfaces operate asynchronously via hardware interrupts and Direct Memory Access (DMA).
    - **Demonstration:** The system computes differential byte rates over time, capturing sudden bursts in I/O and network activity.

    ---

    ### 2. Machine Learning: The Isolation Forest Algorithm

    #### Why Isolation Forest?
    Traditional anomaly detection methods (like distance-based k-NN or density-based LOF) suffer from high computational complexity ($O(N^2)$) in multivariate continuous streams. **Isolation Forest** isolates anomalies instead of profiling normal data points:
    1. **Random Recursive Partitioning:** The algorithm builds an ensemble of isolation trees (iTrees) by randomly selecting a feature and randomly choosing a split value between the feature's minimum and maximum.
    2. **Short Path Length to Isolation:** Anomalies are few and attribute-different, so they require significantly fewer random splits to isolate into leaf nodes compared to dense normal clusters.
    3. **Mathematical Formulation:**
       $$\text{Anomaly Score: } s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
       Where $E(h(x))$ is the average path length of point $x$ over the ensemble of trees, and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree with $n$ nodes:
       $$c(n) = 2 \ln(n - 1) + 0.5772156649 \text{ (Euler's constant)} - \frac{2(n - 1)}{n}$$
    4. **Score Interpretation:**
       - If $s \to 1.0$: Highly anomalous point (isolated with very short tree branches).
       - If $s < 0.5$: Normal point deep inside cluster.

    #### Algorithmic Limitations
    - **Axis-Parallel Splits:** Standard Isolation Forest uses axis-aligned orthogonal cuts, which can produce ghost artifact regions in high dimensions.
    - **Non-Causal:** An anomaly flags statistical uniqueness, not maliciousness. A legitimate software compile or database indexing operation will legitimately trigger an anomaly alert.
    """)


# =====================================================================
# AUTO-REFRESH SCHEDULER
# =====================================================================
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
