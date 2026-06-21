# Data and Environment

Running this project depends on local data and, for some methods, optimization tools.
Expect setup work before experiments run end to end.

## Python Package

The project uses a PyScaffold-style Python package:

- package root: `src/onset/`
- install target: `pip install .`
- focused test command: `python -m unittest tests.te_engine_test`

Dependencies are listed in `pyproject.toml`. Important ones include:

- `networkx`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `highspy`

## External Tools

The simulator expects:

- NetworkX and SciPy for internal ECMP/MCF traffic engineering. SciPy uses its
  bundled open-source HiGHS solver.
- Gurobi and `gurobipy` are only needed for the legacy optimization files
  (`gurobi_doppler.py`, `gurobi_optimization.py`). The active path through
  `open_doppler.py` uses HiGHS and does not require Gurobi.
- Graphviz-style tooling may be needed for `.dot` output and graph rendering.
- SLURM if using the batch scripts directly.

### YATES (external OCaml TE engine) — historical, not required

The `external/yates/` directory contains the OCaml YATES traffic engineering
engine. It is **not required** for the current active pipeline. It was used
historically with TE methods such as `-semimcfraeke` and `-semimcfraekeft`
(Semi-Oblivious MCF with Racke decomposition, akin to SMORE-style traffic
patterns).

The internal TE engine (`src/onset/te/engine.py`) handles `-ecmp` and `-mcf`
and raises a clear error for legacy YATES TE methods. The YATES source is
retained for historical reproducibility of older experiment campaigns.

`src/onset/constants.py` sets `SCRIPT_HOME` to `os.path.abspath(".")`, so many
paths assume commands are run from the repository root.

Run `scripts/check-env.sh` to check Python, optional Gurobi, and core
Python imports. Set `REQUIRE_GUROBI=1` to make missing Gurobi components fail
the check.

Verify the real integration path with:

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_ans_ecmp.py
```

This runs one nonzero ANS traffic matrix through internal baseline ECMP and checks that
MLU is positive, loss is zero, and normalized throughput is one. The locally
verified MLU is `0.804324`.

## Data Layout

Common input locations:

- `data/graphs/gml/`: GML topologies.
- `data/graphs/json/`: JSON topologies, when available.
- `data/graphs/dot/`: DOT topologies for TE evaluation.
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
