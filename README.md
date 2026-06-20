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
### **Prerequisites**
Ensure you have the following installed:
- Python **3.8+**
- Required dependencies (install via `pip`)
- Gurobi (for network optimization)
- YATES (for traffic-engineering evaluation)
- NetworkX, NumPy, Matplotlib

### **YATES**
This simulator invokes the external `yates` command when evaluating traffic
engineering methods. YATES is not vendored into this repository; install it from
[cornell-netlab/yates](https://github.com/cornell-netlab/yates) or initialize
the optional submodule if present:

```bash
git submodule update --init --recursive external/yates
cd external/yates
make && make install
```

If YATES is built locally but not installed on your `PATH`, set:

```bash
export YATES_BIN=/absolute/path/to/yates
```

You can check the local environment with:

```bash
scripts/check-env.sh
```

### **Clone the Repository**
```bash
git clone https://github.com/mattall/topology-programming.git
cd topology-programming
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

*For Gurobi installation, follow the [official guide](https://www.gurobi.com/documentation/).*

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
The **main simulation** can be run using:
```bash
python src/onset/simulator.py <network_name> <number_of_nodes> <experiment_name>
```

### **Example Usage**
```bash
python src/onset/simulator.py example_network 18 experiment1
```
This command:
- Loads the `example_network.gml` topology from `data/graphs/gml/`.
- Assumes the network has 18 nodes.
- Saves results in `data/results/experiment1/`.

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
