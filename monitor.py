"""
System monitoring module for AI-Powered System Resource Manager.
Collects real-time CPU, memory, disk, network, and process metrics using psutil.
Demonstrates Operating System concepts: CPU core scheduling, virtual memory, paging, and I/O.
"""

import time
import datetime
import os
from typing import Dict, Any, List, Optional
import psutil


class SystemMonitor:
    """
    Stateful system monitor that samples OS resource metrics and computes
    delta rates for network and disk I/O.
    """

    def __init__(self):
        # Initialize baseline for non-blocking CPU measurement
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)

        self._last_net_io = psutil.net_io_counters()
        self._last_disk_io = psutil.disk_io_counters()
        self._last_sample_time = time.time()

        # Cache static system specs
        self.logical_cores = psutil.cpu_count(logical=True) or 1
        self.physical_cores = psutil.cpu_count(logical=False) or 1
        self.boot_time = psutil.boot_time()

    def get_root_partition(self) -> str:
        """Determines the appropriate system partition (C:\\ on Windows, / on Unix)."""
        if os.name == "nt":
            return os.getenv("SystemDrive", "C:") + "\\"
        return "/"

    def get_system_snapshot(self) -> Dict[str, Any]:
        """
        Gathers a complete point-in-time snapshot of system hardware and OS metrics.
        Returns a dictionary containing all measured parameters.
        """
        now = time.time()
        dt = max(now - self._last_sample_time, 0.001)

        # 1. CPU Metrics
        cpu_total = psutil.cpu_percent(interval=None)
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        
        cpu_freq_info = psutil.cpu_freq()
        cpu_freq_current = cpu_freq_info.current if cpu_freq_info else 0.0
        cpu_freq_max = cpu_freq_info.max if cpu_freq_info else 0.0

        # CPU Stats (OS scheduling & interrupt awareness)
        try:
            cpu_stats = psutil.cpu_stats()
            ctx_switches = cpu_stats.ctx_switches
            interrupts = cpu_stats.interrupts
        except Exception:
            ctx_switches = 0
            interrupts = 0

        # 2. Virtual Memory (RAM & Swap)
        vmem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        mem_total_gb = round(vmem.total / (1024 ** 3), 2)
        mem_used_gb = round(vmem.used / (1024 ** 3), 2)
        mem_available_gb = round(vmem.available / (1024 ** 3), 2)
        mem_percent = vmem.percent

        swap_total_gb = round(swap.total / (1024 ** 3), 2)
        swap_used_gb = round(swap.used / (1024 ** 3), 2)
        swap_percent = swap.percent

        # 3. Disk Usage
        partition = self.get_root_partition()
        try:
            disk_usage = psutil.disk_usage(partition)
            disk_total_gb = round(disk_usage.total / (1024 ** 3), 2)
            disk_used_gb = round(disk_usage.used / (1024 ** 3), 2)
            disk_free_gb = round(disk_usage.free / (1024 ** 3), 2)
            disk_percent = disk_usage.percent
        except Exception:
            disk_total_gb = 0.0
            disk_used_gb = 0.0
            disk_free_gb = 0.0
            disk_percent = 0.0

        # Disk I/O Rate
        try:
            curr_disk_io = psutil.disk_io_counters()
            if curr_disk_io and self._last_disk_io:
                disk_read_kbps = round((curr_disk_io.read_bytes - self._last_disk_io.read_bytes) / 1024 / dt, 2)
                disk_write_kbps = round((curr_disk_io.write_bytes - self._last_disk_io.write_bytes) / 1024 / dt, 2)
            else:
                disk_read_kbps = 0.0
                disk_write_kbps = 0.0
            self._last_disk_io = curr_disk_io
        except Exception:
            disk_read_kbps = 0.0
            disk_write_kbps = 0.0

        # 4. Network Activity and Rates
        try:
            curr_net_io = psutil.net_io_counters()
            if curr_net_io and self._last_net_io:
                net_sent_kbps = round((curr_net_io.bytes_sent - self._last_net_io.bytes_sent) / 1024 / dt, 2)
                net_recv_kbps = round((curr_net_io.bytes_recv - self._last_net_io.bytes_recv) / 1024 / dt, 2)
            else:
                net_sent_kbps = 0.0
                net_recv_kbps = 0.0
            self._last_net_io = curr_net_io
        except Exception:
            net_sent_kbps = 0.0
            net_recv_kbps = 0.0

        # Update last sample time
        self._last_sample_time = now

        # 5. Process Count
        try:
            pids = psutil.pids()
            process_count = len(pids)
        except Exception:
            process_count = 0

        # 6. Uptime formatting
        uptime_seconds = int(now - self.boot_time)
        uptime_str = str(datetime.timedelta(seconds=uptime_seconds))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "timestamp": timestamp,
            "cpu_percent": float(cpu_total),
            "per_cpu_percent": per_cpu,
            "logical_cores": self.logical_cores,
            "physical_cores": self.physical_cores,
            "cpu_freq_current": cpu_freq_current,
            "cpu_freq_max": cpu_freq_max,
            "context_switches": ctx_switches,
            "interrupts": interrupts,
            "memory_percent": float(mem_percent),
            "memory_total_gb": mem_total_gb,
            "memory_used_gb": mem_used_gb,
            "memory_available_gb": mem_available_gb,
            "swap_percent": float(swap_percent),
            "swap_total_gb": swap_total_gb,
            "swap_used_gb": swap_used_gb,
            "disk_partition": partition,
            "disk_percent": float(disk_percent),
            "disk_total_gb": disk_total_gb,
            "disk_used_gb": disk_used_gb,
            "disk_free_gb": disk_free_gb,
            "disk_read_kbps": max(0.0, disk_read_kbps),
            "disk_write_kbps": max(0.0, disk_write_kbps),
            "net_sent_kb": max(0.0, net_sent_kbps),
            "net_recv_kb": max(0.0, net_recv_kbps),
            "process_count": process_count,
            "uptime_seconds": uptime_seconds,
            "uptime_str": uptime_str,
        }


# Singleton monitor instance for consistent rate measurement across app refreshes
_global_monitor: Optional[SystemMonitor] = None


def get_monitor() -> SystemMonitor:
    """Returns or initializes the singleton SystemMonitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor()
    return _global_monitor


def get_system_snapshot() -> Dict[str, Any]:
    """Helper function to fetch current snapshot from global monitor."""
    return get_monitor().get_system_snapshot()
