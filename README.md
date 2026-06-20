# **Topology Programming Simulator**
_A simulator for reconfigurable optical networks with Optical Topology Programming (OTP)._

📌 **Author:** Matthew Nance-Hall, University of Oregon  
📌 **Purpose:** This simulator evaluates dynamic optical network reconfiguration for improving network resilience and performance.  

---

## **📖 Table of Contents**
- [Introduction](#introduction)
- [Installation](#installation)
- [Input Files](#input-files)
- [Running the Simulator](#running-the-simulator)
- [Understanding the Codebase](#understanding-the-codebase)
- [Contributing](#contributing)
- [License](#license)

---

## **📌 Introduction**
This repository contains a **network simulation tool** for evaluating **Optical Topology Programming (OTP)**, a method that dynamically reconfigures optical networks. 

The simulator models **traffic dynamics**, **network topologies**, and **performance characteristics** under various conditions. It is designed for research, experimentation, and evaluation of optical network behavior.

---

## **🛠 Installation**
### **Clone the Repository**
```bash
git clone https://github.com/mattall/topology-programming.git
cd topology-programming
```

### **Python Environment**

Python 3.8 or newer is required. Most commands assume they are run from the
repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The `TMgen` dependency is currently fetched over GitHub SSH, so package
installation requires working GitHub SSH credentials.

### **Traffic Engineering**

ECMP and MCF routing are implemented in the Python package. ECMP uses
NetworkX shortest paths; MCF uses SciPy's open-source HiGHS linear-programming
backend. Neither method requires YATES, Gurobi, or an external executable.

Check the environment and run the nonzero ANS/ECMP integration smoke test:

```bash
scripts/check-env.sh
PYTHONPATH=src .venv/bin/python scripts/smoke_ans_ecmp.py
```

The tested smoke result has MLU `0.804324`, loss `0.0`, and throughput `1.0`.

### **Gurobi**

Some topology-programming methods still require `gurobipy`, `gurobi_cl`, and a
valid license. Internal ECMP and MCF evaluation do not. Require Gurobi checks
explicitly with:

```bash
REQUIRE_GUROBI=1 scripts/check-env.sh
```

---

## **📂 Input Files**
The simulator requires **three key input files**:

| **File** | **Description** |
|----------|----------------|
| **Topology File (`.gml` or `.json`)** | Defines the network structure (nodes, edges, and capacities). |
| **Traffic Matrix (`.txt`)** | Time-series data defining traffic loads between network nodes. |
| **Host List (`.txt`)** | Lists the active hosts in the network. |

---

### **📌 Example Traffic Matrix (`data/traffic/example_traffic.txt`)**
Each **line represents a time step**, and contains a **flattened matrix** (rows concatenated into a single line).

- The **matrix dimensions** match the number of nodes in the topology file.
- Each entry represents **traffic volume between a source and destination**.

**Example Format:**
```
0.0  100.5  200.3  0.0  50.1  10.0  ...
120.0  0.0  95.3  70.8  0.0  0.0  ...
...
```
- Row 1: **Traffic at time step 1**
- Row 2: **Traffic at time step 2**
- Each value corresponds to a **traffic volume from node i to node j**.

---

## **🚀 Running the Simulator**

For a known-good end-to-end run, start with:

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_ans_ecmp.py
```

The historical command-line wrapper is `src/onset/net_sim.py`. Current research
campaigns are generally launched through scripts under `scripts/Doppler/`,
`scripts/TDSC/`, or `scripts/TNSM/`; read
[`KNOWLEDGE_INDEX.md`](KNOWLEDGE_INDEX.md) before selecting a workflow.

### **Legacy CLI Example**

```bash
PYTHONPATH=src .venv/bin/python src/onset/net_sim.py ANS 18 example \
  -i 1 -te ecmp -t data/traffic/ANS_coremelt_every_link_2.00e+11.txt
```

---

## **📂 Understanding the Codebase**
```
topology-programming/
│── data/                     # Input files
│   ├── graphs/
│   │   ├── gml/              # Network topology files (GML format)
│   │   └── json/             # Network topology files (JSON format)
│   ├── traffic/              # Traffic matrix files
│   └── hosts/                # Host lists
│── results/                  # Output files (logs, analysis)
│── src/                      # Main source code
│   └── onset/
│       ├── simulator.py      # Main simulation script
│       ├── network.py        # Network simulation logic
│       ├── doppler.py        # OTP-based mechanism
│       ├── utils.py          # Helper functions (logging, visualization)
│── tests/                    # Unit tests
│── requirements.txt          # Python dependencies
│── README.md                 # This file
```

---

## **📬 Contributing**
To contribute:
1. **Fork** the repository.
2. **Create a new branch** (`feature-xyz`).
3. **Write clear, documented code**.
4. **Submit a Pull Request**.

---

## **📜 License**
This project is licensed under the **MIT License**.
