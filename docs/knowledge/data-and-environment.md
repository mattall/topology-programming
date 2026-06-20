# Data and Environment

Running this project depends on local data and external optimization/TE tools.
Expect setup work before experiments run end to end.

## Python Package

The project uses a PyScaffold-style Python package:

- package root: `src/onset/`
- install target: `pip install .`
- test command: `pytest`

Dependencies are listed in `setup.cfg`. Important ones include:

- `networkx`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `gurobipy`
- `TMgen`, installed from a Git SSH URL

## External Tools

The simulator expects:

- YATES or compatible traffic-engineering executable access for external TE
  evaluation paths. The expected upstream is
  `https://github.com/cornell-netlab/yates`.
- Gurobi, `gurobipy`, and a working Gurobi license for MCF and
  optimization-backed topology methods. ECMP evaluation does not require it.
- Graphviz-style tooling may be needed for `.dot` output and graph rendering.
- SLURM if using the batch scripts directly.

`src/onset/constants.py` sets `SCRIPT_HOME` to `os.path.abspath(".")`, so many
paths assume commands are run from the repository root.

The simulator resolves YATES as follows:

1. Use `YATES_BIN` if it is set.
2. Otherwise use `yates` from `PATH`.

Run `scripts/check-env.sh` to check Python, YATES, optional Gurobi, and core
Python imports. Set `REQUIRE_GUROBI=1` to make missing Gurobi components fail
the check.

## YATES Submodule

YATES remains a separate project pinned as a submodule under:

```text
external/yates
```

The upstream dependency bounds are not sufficient for a reproducible build
against current opam repositories. Use the tested setup script:

```bash
scripts/setup-yates.sh
export YATES_BIN="$(opam exec --switch=yates -- which yates)"
```

The script requires opam 2.2+, `pkgconf`, Git, Make, and a C toolchain. It:

1. Initializes `external/yates`.
2. Creates an isolated OCaml 4.12 opam switch named `yates`.
3. Uses Core 0.14.1, Async 0.14.0, PPX Jane 0.14.0, and TCP/IP 7.1.2.
4. Checks out Frenetic commit `fdce6cc` under ignored `.cache/` storage.
5. Applies `scripts/patches/frenetic-tcpip-checksum.patch`.
6. Builds and installs YATES into the switch.

Verify the real integration path with:

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_ans_ecmp.py
```

This runs one nonzero ANS traffic matrix through baseline ECMP and checks that
MLU is positive, loss is zero, and normalized throughput is one. The locally
verified MLU is `0.804324`.

## Data Layout

Common input locations:

- `data/graphs/gml/`: GML topologies.
- `data/graphs/json/`: JSON topologies, when available.
- `data/graphs/dot/`: DOT topologies for Yates-style runs.
- `data/traffic/`: flattened traffic matrices.
- `data/hosts/`: host files.
- `data/txp/`: transponder allocation data.
- `data/paths/`: path data and optimization path caches.

Common output locations:

- `data/results/`: simulator result directories.
- `data/reports/`: compact CSV summaries.
- `data/plots/`: plot outputs.
- `logs/`: logs and infeasible-model artifacts when present.
- `.temp/`: temporary files, including hashed simulator nonce paths.

## Traffic Matrix Format

Traffic matrices are flattened square matrices, one row per time step. For a
network with `n` hosts, each row has `n * n` values. Scripts commonly expect
names such as:

```text
data/traffic/background_Comcast-tm
data/traffic/background_CEN-tm
data/traffic/background_Gigapop-tm
```

The Doppler runner constructs this path as:

```python
f"data/traffic/{traffic}_{network}-tm"
```

## Result Files

Core result files include:

- `MaxExpCongestionVsIterations.dat`
- `CongestionLossVsIterations.dat`
- `TotalThroughputVsIterations.dat`
- `MeanCongestionVsIterations.dat`
- `k90ExpCongestionVsIterations.dat`
- `NumPathsVsIterations.dat`

Doppler-specific files include:

- `TotalSolutions.dat`
- `DopplerMinMLU.dat`
- `DopplerMLU.dat`
- `OptTime.dat`
- `CurrTopoID.dat`
- `OptimalTopoID.dat`

## Maintenance Notes

- `Makefile clean` deletes generated result/report/temp data. Review it before
  running.
- Some scripts contain absolute paths or cluster-specific assumptions.
- Some tests depend on data files that may not exist in a fresh clone.
- `scripts/Doppler/experiment_params.py` likely needs `"Gigapop"` fixed in the
  `hosts` map if using the latest non-debug `exp-v2.py` defaults.
