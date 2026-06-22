# Topology Programming Simulator

A simulator for reconfigurable optical networks with Optical Topology
Programming (OTP).

## Installation

The minimum supported version is Python 3.13. All commands assume they are run
from the repository root.

```bash
git clone --recurse-submodules https://github.com/mattall/topology-programming.git
cd topology-programming
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[testing,dev]"
```

`TMgen` (traffic-matrix generation) is vendored as a git submodule at
`vendor/TMgen`.  If you need attack-traffic generation, install it:

```bash
pip install -e vendor/TMgen
```

## Traffic Engineering

ECMP, MCF, and SMORE (semi-oblivious MCF with Raecke decomposition)
routing are built-in.  ECMP uses NetworkX shortest paths; MCF uses
SciPy's open-source HiGHS solver; SMORE uses FRT tree decomposition
with multiplicative weights for path selection and a restricted-MCF
LP for rate adaptation.  No external executables are required.

## Topology Programming

All four MILP-based topology-programming methods run on the open-source
HiGHS backend (via `highspy`).  No Gurobi license or `gurobipy`
installation is needed.

| Method string | Description |
|---|---|
| `baseline`    | Evaluate ECMP/MCF without changing the input topology |
| `doppler`     | Doppler reconnaissance defense (TNSM 2024) |
| `onset_v3`    | ONSET DDoS defense — edge-flow formulation (TDSC 2025) |
| `onset_v2`    | ONSET DDoS defense — path-flow formulation (TDSC 2025) |
| `onset`       | Original topology programming formulation |
| `OTP`         | Offline Traffic Provisioning — shortcut-link heuristic |
| `greylambda`  | Add circuits on fully-congested edges |
| `cache`       | Cache-based defense (`Defender` module) |
| `BVT`         | Bandwidth-variable transceiver emulation |
| `TBE`         | Temporary bandwidth expansion during flashcrowd |

Method dispatch is data-driven through `src/onset/method_registry.py`.

## Check the environment and smoke-test

```bash
scripts/check-env.sh
python scripts/smoke_ans_ecmp.py
```

The smoke test creates its topology and traffic matrix in a temporary directory;
it does not require repository experiment data.

The same checks used by CI are:

```bash
ruff check
ruff format --check
mypy
pytest
scripts/check-env.sh
python scripts/smoke_ans_ecmp.py
```

## Maintained and historical code

The maintained Python surface is the `Simulation.perform_sim` pipeline, method
registry and handlers, internal ECMP/MCF engine, HiGHS topology solvers,
preprocessing, reporting, and their small runtime helpers. The exact lint and
type-check paths are listed in `pyproject.toml`, so local, pre-commit, tox, and
CI runs agree.

Historical campaign launchers under `scripts/Doppler`, `scripts/TDSC`, and
`scripts/TNSM`, plotting/analysis utilities, vendored code, and the legacy
Gurobi modules are retained for reproducibility but are not part of the
maintained static-analysis surface. Gurobi remains optional and is not used by
the maintained pipeline.

## Running experiments

Research campaigns are launched through scripts under `scripts/Doppler/`,
`scripts/TDSC/`, or `scripts/TNSM/`.  Read [`KNOWLEDGE_INDEX.md`](KNOWLEDGE_INDEX.md)
before selecting a workflow.

## Codebase overview

```
src/onset/
├── simulator.py            # Simulation class, perform_sim loop, dispatch
├── handlers.py             # Method handler functions (_run_milp_method, _run_otp, ...)
├── validation.py           # Input validation, topology/traffic/host verification
├── method_registry.py      # MethodConfig, _METHOD_REGISTRY, _resolve_method
├── reporter.py             # write_optimization_reports, evaluate_candidate_topologies
├── open_doppler.py         # HiGHS MILP builders and solvers (all four methods)
├── preprocessing.py        # build_optimization_problem factory
├── base_types.py           # OptimizationProblem, TopologySolution, OptimizationResult
├── attacker.py             # DDoS attack generation (lazy TMgen dependency)
├── te/
│   └── engine.py           # ECMP, MCF, and SMORE/Raecke traffic engineering
└── utilities/              # Graph I/O, plotting, logging, path utilities
```

## License

MIT
