# AI-Powered System Resource Manager

An intelligent desktop operating system resource monitoring and analysis application built with **Python**, **Streamlit**, **psutil**, **Scikit-Learn**, **Plotly**, and **SQLite**.

The project monitors system resources in real time, applies **Isolation Forest** machine learning to detect multidimensional performance anomalies, provides deterministic rule-based recommendations grounded in core Operating System concepts, and offers non-destructive process inspection and management.

---

## 1. Project Overview & Problem Statement

Modern desktop operating systems execute hundreds of concurrent tasks, threads, and background daemons. When system slowdowns or high resource consumption occur, ordinary users and students often lack visibility into:
1. Whether elevated utilization is concentrated in a single process or distributed across many tasks.
2. The exact operating system subsystem under pressure (CPU ready-queue contention, virtual memory page thrashing, storage exhaustion, or socket I/O saturation).
3. Whether unusual resource combinations represent statistical anomalies or standard operational spikes.

Traditional Task Managers provide static tabular numbers without machine-learning insights, correlation across subsystems, or pedagogical Operating System explanations.

**AI-Powered System Resource Manager** addresses this problem by:
- Collecting live hardware and kernel telemetry via `psutil`.
- Applying an **Isolation Forest** ensemble model to evaluate multivariate anomalies across CPU, RAM, Disk, Network, and Process Count.
- Explaining detected anomalies through Z-score feature attribution rather than unhelpful, alarmist claims.
- Providing deterministic OS heuristic recommendations.
- Logging telemetry and detected anomalies in persistent **SQLite** storage with maintenance pruning and CSV export.

---

## 2. Operating System Concepts Demonstrated

The application serves as a practical, working demonstration of fundamental undergraduate Operating Systems principles:

### A. CPU Scheduling & Ready Queue Contention
- **Theory:** The operating system kernel scheduler (e.g., Windows Multilevel Feedback Queue with priority boosts, Linux Completely Fair Scheduler) allocates discrete CPU time slices (quanta) to runnable threads in the ready queue.
- **Implementation:** The application captures total CPU utilization and per-core scheduling distribution. It diagnoses:
  - **Dominant Process Saturation:** When a single process consumes excessive CPU time, starving other threads and increasing scheduler dispatch latency.
  - **Thread Contention:** When aggregate CPU load is high across dozens of processes, leading to frequent context switches, Translation Lookaside Buffer (TLB) flushes, and CPU cache misses.
  - **Single-Thread Bottleneck:** When one core is pegged at 100% while other cores are idle, indicating an application lacking multi-threading parallelism.

### B. Process Control Blocks (PCB) & Safe Lifecycle Management
- **Theory:** The OS kernel maintains a Process Control Block (PCB) for each process, storing its Process ID (PID), state, scheduling priority, thread count, memory allocation pointers, and open file descriptors.
- **Implementation:** The **Process Manager** module inspects live PCB data via system calls without requiring administrator privileges. Users can search, sort by CPU/RAM/threads, inspect process details, and terminate selected non-critical processes.
- **Safety Guarantee:** Critical OS processes (e.g., `System`, `smss.exe`, `csrss.exe`, `wininit.exe`, `services.exe`, `init`, PID $\le$ 4) are strictly protected against accidental termination. All termination actions require explicit user confirmation.

### C. Virtual Memory Management & Page Thrashing
- **Theory:** Operating systems provide virtual memory via hardware paging and page tables. When the working sets of active processes exceed physical RAM, the Virtual Memory Manager (VMM) evicts pages to secondary storage (the Windows pagefile or Unix swap partition).
- **Implementation:** The dashboard tracks physical RAM allocation, available frames, and pagefile/swap usage. When physical RAM exceeds 90% and swap usage rises, the engine issues a **page thrashing** warning, identifying the top memory consumer.

### D. I/O Subsystems & Network Socket Activity
- **Theory:** Disk and network communications operate asynchronously via hardware interrupts, Direct Memory Access (DMA), and buffer caches.
- **Implementation:** The monitor computes differential throughput rates ($\text{KB/s}$) across sampling intervals, highlighting sustained I/O saturation that may degrade interactive desktop responsiveness.

---

## 3. Machine Learning: The Isolation Forest Component

### Why Isolation Forest?
Traditional anomaly detection algorithms (such as density-based Local Outlier Factor or distance-based k-Means/k-NN) require computing pairwise distances ($O(N^2)$), making them computationally heavy for real-time streaming telemetry. 

The **Isolation Forest** algorithm (Liu, Ting, and Zhou, 2008) operates on the principle that **anomalies are 'few and different'**. Rather than building a profile of normal instances, it isolates anomalous observations through randomized recursive partitioning:

1. **Isolation Trees (iTrees):** An ensemble of binary trees is constructed. At each node, a feature is selected at random, and a random split value is chosen between that feature's minimum and maximum values.
2. **Path Length Property:** Normal, dense points require many partitions to be isolated deep in the tree. Anomalous points with unusual multivariate combinations are isolated close to the root with noticeably shorter path lengths $h(x)$.
3. **Scoring Formulation:**
   $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
   Where:
   - $E(h(x))$ is the average path length of point $x$ across all trees in the ensemble.
   - $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree with $n$ nodes:
     $$c(n) = 2 \ln(n - 1) + 0.5772156649 \text{ (Euler's constant)} - \frac{2(n - 1)}{n}$$
   - $s \to 1.0$: Confirmed anomaly.
   - $s < 0.5$: Well within the normal operating cluster.

### Feature Space
The model evaluates a 6-dimensional resource vector:
1. `cpu_percent` (0% - 100%)
2. `memory_percent` (0% - 100%)
3. `disk_percent` (0% - 100%)
4. `net_sent_kb` (Upload rate, KB/s)
5. `net_recv_kb` (Download rate, KB/s)
6. `process_count` (Active process table entries)

### Transparent Baseline & Online Retraining
- **Initial Baseline:** The model is bootstrapped with a realistic synthetic desktop baseline dataset (600 samples) modeling typical multi-modal desktop behavior (idle, web browsing, and heavy compute modes).
- **Live Retraining:** As real system snapshots accumulate in SQLite ($\ge 25$ samples), the user can retrain the model directly on their actual hardware workload using the sidebar control.
- **Feature Attribution via Z-Scores:** When an observation is flagged as an anomaly, the application calculates the deviation of each feature relative to baseline parameters ($Z = \frac{x - \mu}{\sigma}$), pinpointing the exact metrics driving the alert.
- **Ethical & Technical Caveat:** Anomalies denote statistical deviations, not malware or hardware failure. Heavy batch compiling or 4K rendering naturally generates anomalous profiles without being malicious.

---

## 4. System Architecture & Module Structure

```
system_resource_manager/
├── app.py                      # Streamlit frontend dashboard (tabs, charts, tables)
├── monitor.py                  # Real-time OS metric collector (psutil wrapper)
├── process_manager.py          # Process enumeration, sorting, safety checks & termination
├── anomaly_detector.py         # Isolation Forest model, feature scaling & Z-score attribution
├── recommendation_engine.py    # Deterministic rule-based OS heuristic analysis
├── database.py                 # SQLite storage, time-series querying, pruning & schema
├── requirements.txt            # Python package dependencies
├── README.md                   # Complete architectural and operational documentation
└── tests/
    └── test_core.py            # Automated test suite with mocks
```

### Module Responsibilities

| Module | Core Responsibility |
|---|---|
| `app.py` | Renders the responsive Streamlit dashboard across 6 tabs, coordinates auto-refresh, and handles interactive user confirmation dialogs. |
| `monitor.py` | Queries kernel telemetry via `psutil`, calculates differential network/disk rates, and gathers per-core scheduling load. |
| `process_manager.py` | Enumerates processes, parses PCB details, enforces protected system process guards, and safely terminates targeted PIDs. |
| `anomaly_detector.py` | Fits Scikit-Learn `IsolationForest` on baseline or live data, computes anomaly probabilities, and builds technical diagnostic explanations. |
| `recommendation_engine.py` | Applies deterministic OS rules to detect CPU saturation, page thrashing, storage depletion, and context-switching overhead. |
| `database.py` | Manages SQLite connection pooling, tables (`system_metrics`, `anomaly_events`), chronological queries, and retention pruning. |

---

## 5. Technology Stack

- **Language:** Python 3.10+
- **Frontend Dashboard:** Streamlit
- **System Telemetry:** psutil
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`IsolationForest`, `StandardScaler`)
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
- **Storage:** SQLite3 (standard library)
- **Testing:** Pytest with Unittest Mocks

---

## 6. Installation & Execution (Windows Instructions)

Follow these exact steps in **PowerShell** or **Command Prompt** to set up and run the application.

### Step 1: Open PowerShell in the Project Directory
```powershell
cd "c:\Users\sahithi reddy\Downloads\system_resource_manager"
```

### Step 2: Create and Activate a Virtual Environment
```powershell
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once).*

### Step 3: Install Required Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 4: Run the Test Suite
Verify that all modules, database operations, and safety guards pass:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Step 5: Launch the Application
Start the Streamlit dashboard:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Streamlit will launch and automatically open your default browser at `http://localhost:8501`.

---

## 7. Testing & Validation Strategy

The test suite in [`tests/test_core.py`](file:///c:/Users/sahithi%20reddy/Downloads/system_resource_manager/tests/test_core.py) executes deterministic tests independent of the host machine's live load:

1. **Database Layer:** Tests table creation, metric insertion, anomaly logging, chronological querying, record pruning, and history clearing.
2. **System Monitor:** Tests snapshot structure, value boundary validations ($0\% \le \text{CPU, RAM, Disk} \le 100\%$), and mocked hardware responses.
3. **Process Manager Safety:** Verifies retrieval of PCB metrics, search filtering, and confirms that attempts to terminate protected system processes (PID 0, 4, `csrss.exe`, `System`) are blocked.
4. **Machine Learning Anomaly Detection:** Validates synthetic baseline generation, model training, scoring of normal vs. extreme outlier snapshots, and retraining on historical DataFrames.
5. **Recommendation Engine:** Tests deterministic heuristics under normal, high-CPU saturation, and severe memory thrashing conditions.

---

## 8. Expected Output & UI Demonstration

### Tab 1: Overview
- **Metric Cards:** Real-time CPU, RAM, Disk, Network Throughput, and Active Process count with live deltas.
- **Live Trend Chart:** Plotly line chart plotting continuous CPU, RAM, and Disk utilization over time.
- **CPU Scheduling Breakdown:** Bar chart displaying the workload distribution across each logical CPU core.
- **Memory & Swap Breakdown:** Donut chart contrasting used RAM, available RAM, and swap page allocation.

### Tab 2: Process Manager
- **Process Table:** Searchable by PID or name, sortable by CPU %, Memory %, RAM (MB), or thread count.
- **Safe Termination Section:** Process inspection panel with state, thread count, user account, and an explicit confirmation checkbox before sending termination signals.

### Tab 3: AI Anomaly Analysis
- **Anomaly Confidence Meter:** Radial gauge displaying the current anomaly score (0% - 100%).
- **Technical Narrative:** Nuanced explanation distinguishing statistical anomaly signatures from software defects.
- **Feature Deviation Table & Bar Chart:** Z-score departure ($σ$) for each monitored resource attribute.

### Tab 4: Intelligent Recommendations
- Categorized actionable cards (`CPU Scheduling & Execution`, `Virtual Memory & Paging`, `File System & Storage Capacity`) with severity tags and distinct heuristic labels.

### Tab 5: Resource History
- Chronological time-series graphs, raw telemetry records, detected anomaly logs, database pruning controls, and CSV download.

### Tab 6: OS Concepts & Architecture
- Academic documentation explaining CPU scheduling ready queues, PCB attributes, two-level page tables, thrashing, and the mathematical formulation of Isolation Forest.

---

## 9. Limitations & Ethical Design Constraints

1. **Permissions:** Normal monitoring functions without administrator privileges. However, terminating processes owned by other user accounts or elevated services will return an `AccessDenied` notice.
2. **Heuristic vs. Causal:** The Isolation Forest detects statistical anomalies, not root causes or security threats. Recommendations are advisory and do not automatically modify system configuration.
3. **Storage Bound:** SQLite metrics are periodically pruned to prevent unbounded disk growth.
