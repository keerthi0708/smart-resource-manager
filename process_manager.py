"""
Process Manager module for AI-Powered System Resource Manager.
Provides inspection, search, sorting, and safe termination of running OS processes.
Demonstrates Operating System concepts: Process Control Blocks (PCB), process states,
thread counts, and memory/CPU resource consumption.
"""

import os
import datetime
from typing import Dict, Any, List, Optional, Tuple
import psutil
import pandas as pd

# Set of critical OS process names and low PIDs that must NEVER be terminated
PROTECTED_PROCESS_NAMES = {
    "system", "system idle process", "registry", "smss.exe", "csrss.exe",
    "wininit.exe", "services.exe", "lsass.exe", "svchost.exe", "winlogon.exe",
    "init", "systemd", "launchd", "kernel_task"
}
PROTECTED_PIDS = {0, 1, 4}


def get_processes(
    sort_by: str = "cpu_percent",
    ascending: bool = False,
    limit: Optional[int] = 100,
    search: str = ""
) -> pd.DataFrame:
    """
    Safely iterates over running processes, collecting key PCB metrics.
    Filters and sorts the result cleanly without crashing on permission errors.
    """
    proc_list: List[Dict[str, Any]] = []

    for proc in psutil.process_iter([
        "pid", "name", "username", "cpu_percent", "memory_percent",
        "memory_info", "status", "num_threads", "create_time"
    ]):
        try:
            info = proc.info
            pid = info["pid"]
            name = info["name"] or "Unknown"

            # Filter by search string if provided
            if search:
                search_lower = search.lower().strip()
                if search_lower not in name.lower() and search_lower not in str(pid):
                    continue

            # Compute resident memory in MB
            mem_info = info.get("memory_info")
            rss_mb = round(mem_info.rss / (1024 * 1024), 2) if mem_info else 0.0

            # Format process creation time
            ctime = info.get("create_time")
            created_str = (
                datetime.datetime.fromtimestamp(ctime).strftime("%H:%M:%S")
                if ctime else "N/A"
            )

            cpu_pct = float(info.get("cpu_percent") or 0.0)
            mem_pct = round(float(info.get("memory_percent") or 0.0), 2)
            threads = int(info.get("num_threads") or 1)
            status = str(info.get("status") or "unknown")
            user = str(info.get("username") or "N/A")

            proc_list.append({
                "pid": pid,
                "name": name,
                "cpu_percent": cpu_pct,
                "memory_percent": mem_pct,
                "memory_mb": rss_mb,
                "num_threads": threads,
                "status": status,
                "username": user,
                "created_time": created_str,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process terminated or protected during iteration
            continue

    df = pd.DataFrame(proc_list)
    if df.empty:
        return pd.DataFrame(columns=[
            "pid", "name", "cpu_percent", "memory_percent",
            "memory_mb", "num_threads", "status", "username", "created_time"
        ])

    # Validate sort column
    valid_sort_col = sort_by if sort_by in df.columns else "cpu_percent"
    df = df.sort_values(by=valid_sort_col, ascending=ascending)

    if limit and limit > 0:
        df = df.head(limit)

    return df.reset_index(drop=True)


def get_process_details(pid: int) -> Optional[Dict[str, Any]]:
    """
    Fetches detailed information for a specific process PID before confirmation.
    Returns None if the process does not exist or access is completely denied.
    """
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            name = proc.name()
            status = proc.status()
            cpu_pct = proc.cpu_percent(interval=0.05)
            mem_info = proc.memory_info()
            rss_mb = round(mem_info.rss / (1024 * 1024), 2)
            mem_pct = round(proc.memory_percent(), 2)
            threads = proc.num_threads()
            
            try:
                username = proc.username()
            except (psutil.AccessDenied, Exception):
                username = "System / Protected"

            try:
                cmdline = " ".join(proc.cmdline())
            except (psutil.AccessDenied, Exception):
                cmdline = "Access Denied"

            return {
                "pid": pid,
                "name": name,
                "status": status,
                "cpu_percent": cpu_pct,
                "memory_mb": rss_mb,
                "memory_percent": mem_pct,
                "num_threads": threads,
                "username": username,
                "cmdline": cmdline,
                "is_protected": is_protected_process(pid, name)
            }
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def is_protected_process(pid: int, name: str) -> bool:
    """
    Checks whether a process is a critical system component that should not be killed.
    """
    if pid in PROTECTED_PIDS or pid <= 4:
        return True
    if name.lower() in PROTECTED_PROCESS_NAMES:
        return True
    return False


def terminate_process(pid: int) -> Tuple[bool, str]:
    """
    Safely terminates a process with PID.
    Guards against killing critical system processes and handles access permission errors.
    Returns (success: bool, status_message: str).
    """
    if not isinstance(pid, int) or pid <= 0:
        return False, f"Invalid PID: {pid}. PID must be a positive integer."

    try:
        proc = psutil.Process(pid)
        proc_name = proc.name()

        # Guard critical operating system processes
        if is_protected_process(pid, proc_name):
            return False, f"Safety block: '{proc_name}' (PID {pid}) is a critical operating system component and cannot be terminated."

        # Attempt graceful termination first
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
            return True, f"Process '{proc_name}' (PID {pid}) was successfully terminated."
        except psutil.TimeoutExpired:
            # If still alive after 2 seconds, force kill
            proc.kill()
            proc.wait(timeout=1.0)
            return True, f"Process '{proc_name}' (PID {pid}) did not exit gracefully and was forcefully stopped."

    except psutil.NoSuchProcess:
        return False, f"Process with PID {pid} no longer exists."
    except psutil.AccessDenied:
        return False, f"Access Denied: Insufficient OS privileges to terminate PID {pid}. Run as administrator if required."
    except Exception as e:
        return False, f"Unexpected error while attempting to terminate PID {pid}: {str(e)}"
