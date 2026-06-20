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

- Gurobi, `gurobipy`, and a working Gurobi license.
- YATES or compatible traffic-engineering executable access for external TE
  evaluation paths. The expected upstream is
  `https://github.com/cornell-netlab/yates`.
- Graphviz-style tooling may be needed for `.dot` output and graph rendering.
- SLURM if using the batch scripts directly.

`src/onset/constants.py` sets `SCRIPT_HOME` to `os.path.abspath(".")`, so many
paths assume commands are run from the repository root.

The simulator resolves YATES as follows:

1. Use `YATES_BIN` if it is set.
2. Otherwise use `yates` from `PATH`.

Run `scripts/check-env.sh` to check Python, YATES, Gurobi, and core Python
imports.

## Optional YATES Submodule

YATES should remain a separate project, but this repo can pin it as an optional
submodule under:

```text
external/yates
```

Initialize and build it with:

```bash
git submodule update --init --recursive external/yates
cd external/yates
make && make install
```

If you prefer not to install globally, build YATES and then set:

```bash
export YATES_BIN=/absolute/path/to/yates
```

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

- The top-level `README.md` replaced `README.rst`, but `setup.cfg` still points
  `long_description` at `README.rst`.
- `Makefile clean` deletes generated result/report/temp data. Review it before
  running.
- Some scripts contain absolute paths or cluster-specific assumptions.
- Some tests depend on data files that may not exist in a fresh clone.
- `scripts/Doppler/experiment_params.py` likely needs `"Gigapop"` fixed in the
  `hosts` map if using the latest non-debug `exp-v2.py` defaults.
