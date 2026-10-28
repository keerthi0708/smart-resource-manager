"""
Recommendation Engine module for AI-Powered System Resource Manager.
Generates clear, rule-based diagnostic recommendations grounded in Operating System concepts
such as CPU scheduling contention, virtual memory paging, disk I/O thrashing, and process bloat.
Distinct from the ML Isolation Forest anomaly detector.
"""

from typing import Dict, Any, List, Optional
import pandas as pd


def generate_recommendations(
    snapshot: Dict[str, Any],
    top_processes: Optional[pd.DataFrame] = None,
    history_df: Optional[pd.DataFrame] = None,
    ml_result: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates current system metrics, recent trends, and running processes against
    principled OS thresholds to yield actionable, non-destructive recommendations.
    """
    recommendations: List[Dict[str, Any]] = []

    cpu_pct = snapshot.get("cpu_percent", 0.0)
    mem_pct = snapshot.get("memory_percent", 0.0)
    swap_pct = snapshot.get("swap_percent", 0.0)
    disk_pct = snapshot.get("disk_percent", 0.0)
    net_sent_kb = snapshot.get("net_sent_kb", 0.0)
    net_recv_kb = snapshot.get("net_recv_kb", 0.0)
    proc_count = snapshot.get("process_count", 0)
    per_cpu = snapshot.get("per_cpu_percent", [])

    # Extract top process by CPU and memory if available
    top_cpu_proc = None
    top_mem_proc = None

    if top_processes is not None and not top_processes.empty:
        # Sort by CPU
        by_cpu = top_processes.sort_values(by="cpu_percent", ascending=False)
        if not by_cpu.empty:
            top_cpu_proc = by_cpu.iloc[0].to_dict()

        # Sort by Memory
        by_mem = top_processes.sort_values(by="memory_percent", ascending=False)
        if not by_mem.empty:
            top_mem_proc = by_mem.iloc[0].to_dict()

    # 1. CPU Scheduling & Contention Rules
    if cpu_pct >= 85.0:
        if top_cpu_proc and top_cpu_proc.get("cpu_percent", 0) >= 40.0:
            recommendations.append({
                "category": "CPU Scheduling & Execution",
                "severity": "CRITICAL",
                "title": f"Dominant Process CPU Saturation ({top_cpu_proc.get('name')})",
                "description": (
                    f"Overall CPU load is {cpu_pct}%. Process '{top_cpu_proc.get('name')}' "
                    f"(PID: {top_cpu_proc.get('pid')}) is consuming {top_cpu_proc.get('cpu_percent')}% of CPU time, "
                    "monopolizing scheduling quantum slices and starving interactive threads."
                ),
                "action": f"Review PID {top_cpu_proc.get('pid')} in the Process Manager. If unresponsive, consider terminating it.",
                "rule_type": "Deterministic Rule-Based Heuristic"
            })
        else:
            recommendations.append({
                "category": "CPU Scheduling & Execution",
                "severity": "WARNING",
                "title": "Multi-Process Aggregate CPU Contention",
                "description": (
                    f"CPU utilization is elevated at {cpu_pct}%, distributed across numerous concurrent tasks. "
                    "High thread concurrency triggers frequent kernel context switches and CPU cache invalidations."
                ),
                "action": "Close unnecessary background utilities and browser tabs to reduce ready-queue scheduling depth.",
                "rule_type": "Deterministic Rule-Based Heuristic"
            })
    elif cpu_pct >= 70.0:
        recommendations.append({
            "category": "CPU Scheduling & Execution",
            "severity": "WARNING",
            "title": "Moderate CPU Load",
            "description": f"CPU utilization is at {cpu_pct}%. The operating system scheduler has fewer idle cycles available for bursts.",
            "action": "Monitor long-running tasks to prevent sustained thermal throttling.",
            "rule_type": "Deterministic Rule-Based Heuristic"
        })

    # Core Imbalance Check (Single-Threaded Bottleneck)
    if per_cpu and len(per_cpu) > 1:
        max_core = max(per_cpu)
        min_core = min(per_cpu)
        if max_core >= 95.0 and min_core <= 20.0 and cpu_pct < 60.0:
            recommendations.append({
                "category": "CPU Core Scheduling Asymmetry",
                "severity": "INFO",
                "title": "Single-Threaded Execution Bottleneck",
                "description": (
                    f"One logical core is pegged at {max_core}%, while other cores remain below {min_core}%. "
                    "An active application is bound by a single execution thread and cannot exploit multi-core parallelism."
                ),
                "action": "Ensure compute-heavy tasks utilize multi-threading or asynchronous worker pools where supported.",
                "rule_type": "Deterministic Rule-Based Heuristic"
            })

    # 2. Virtual Memory & Paging Rules
    if mem_pct >= 90.0:
        mem_detail = ""
        if top_mem_proc:
            mem_detail = (
                f" Primary consumer: '{top_mem_proc.get('name')}' (PID {top_mem_proc.get('pid')}) "
                f"allocating {top_mem_proc.get('memory_mb')} MB ({top_mem_proc.get('memory_percent')}%)."
            )
        recommendations.append({
            "category": "Virtual Memory & Paging",
            "severity": "CRITICAL",
            "title": "Severe Physical Memory Pressure (Thrashing Risk)",
            "description": (
                f"RAM usage has reached {mem_pct}%. Available physical frames are nearly depleted.{mem_detail} "
                "The OS Virtual Memory Manager will be forced to page active working sets to disk, "
                "inducing severe I/O thrashing and UI freezes."
            ),
            "action": "Immediately close high-memory applications in the Process Manager to release resident memory pages.",
            "rule_type": "Deterministic Rule-Based Heuristic"
        })
    elif mem_pct >= 75.0:
        recommendations.append({
            "category": "Virtual Memory & Paging",
            "severity": "WARNING",
            "title": "Elevated Memory Utilization",
            "description": f"Physical RAM is {mem_pct}% full. Working set cache headroom is restricted.",
            "action": "Close idle programs before launching memory-intensive workloads.",
            "rule_type": "Deterministic Rule-Based Heuristic"
        })

    # Swap File / Pagefile warning
    if swap_pct >= 60.0:
        recommendations.append({
            "category": "Virtual Memory & Paging",
            "severity": "WARNING",
            "title": "Heavy Swap / Pagefile Utilization",
            "description": (
                f"System paging space is {swap_pct}% utilized ({snapshot.get('swap_used_gb', 0)} GB). "
                "Frequent secondary storage page-ins degrade execution throughput."
            ),
            "action": "Reduce open process footprint to minimize secondary storage paging overhead.",
            "rule_type": "Deterministic Rule-Based Heuristic"
        })

    # 3. Storage Capacity Rules
    if disk_pct >= 90.0:
        recommendations.append({
            "category": "File System & Storage Capacity",
            "severity": "CRITICAL",
            "title": "Critical Storage Depletion on System Drive",
            "description": (
                f"Root volume ({snapshot.get('disk_partition', 'System')}) has only "
                f"{snapshot.get('disk_free_gb', 0)} GB free ({disk_pct}% full). "
                "The OS requires free sectors for hibernation files, crash dumps, and swap growth."
            ),
            "action": "Clear temporary directories, recycle bins, or relocate large archives to prevent system lockup.",
            "rule_type": "Deterministic Rule-Based Heuristic"
        })
    elif disk_pct >= 80.0:
        recommendations.append({
            "category": "File System & Storage Capacity",
            "severity": "INFO",
            "title": "Storage Volume Over 80% Capacity",
            "description": f"Root volume is {disk_pct}% utilized. {snapshot.get('disk_free_gb', 0)} GB remaining.",
            "action": "Plan maintenance cleanup to retain sufficient buffer for system updates and cache files.",
            "rule_type": "Deterministic Rule-Based Heuristic"
        })

    # 4. Network & I/O Activity Rules
    total_net_rate = net_sent_kb + net_recv_kb
    if total_net_rate >= 10240.0:  # > 10 MB/s
        mb_rate = round(total_net_rate / 1024, 2)
        recommendations.append({
            "category": "Network & I/O Bandwidth",
            "severity": "INFO",
            "title": f"High Network Throughput ({mb_rate} MB/s)",
            "description": (
                f"Current network data rate is {mb_rate} MB/s "
                f"(Down: {round(net_recv_kb / 1024, 2)} MB/s, Up: {round(net_sent_kb / 1024, 2)} MB/s). "
                "Intensive network socket I/O increases network stack interrupt processing."
            ),
            "action": "Verify if active downloads, streaming, cloud synchronization, or backups are expected.",
            "rule_type": "Deterministic Rule-Based Heuristic"
        })

    # 5. Process Table Density Rules
    if proc_count >= 250:
        recommendations.append({
            "category": "Process Management & OS Overhead",
            "severity": "INFO",
            "title": f"High Process Density ({proc_count} Active Processes)",
            "description": (
                f"There are currently {proc_count} processes in the OS process table. "
                "While modern kernels schedule tasks efficiently, each process retains thread structures, "
                "handles, and page table entries in kernel memory."
            ),
            "action": "Audit startup applications and disable unnecessary background daemons or helper services.",
            "rule_type": "Deterministic Rule-Based Heuristic"
        })

    # 6. Fallback Healthy Status
    if not recommendations:
        recommendations.append({
            "category": "System Health Overview",
            "severity": "SUCCESS",
            "title": "System Operating in Optimal Balance",
            "description": (
                f"CPU utilization ({cpu_pct}%), Memory ({mem_pct}%), and Disk usage ({disk_pct}%) "
                "are well within nominal operating thresholds. CPU scheduling ready queues are clear, "
                "and no memory pressure or I/O starvation is observed."
            ),
            "action": "No immediate user action required. System resources are operating smoothly.",
            "rule_type": "Deterministic Rule-Based Heuristic"
        })

    return recommendations
