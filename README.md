# Topology Programming Simulator

A simulator for reconfigurable optical networks with Optical Topology
Programming (OTP).

## Installation

Python 3.11 or newer is required.  All commands assume they are run from
the repository root.

```bash
git clone --recurse-submodules https://github.com/mattall/topology-programming.git
cd topology-programming
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

`TMgen` (traffic-matrix generation) is vendored as a git submodule at
`vendor/TMgen`.  If you need attack-traffic generation, install it:

```bash
pip install -e vendor/TMgen
```

## Traffic Engineering

ECMP and MCF routing are built-in.  ECMP uses NetworkX shortest paths;
MCF uses SciPy's open-source HiGHS solver.  No external executables are
required.

## Topology Programming

All four MILP-based topology-programming methods run on the open-source
HiGHS backend (via `highspy`).  No Gurobi license or `gurobipy`
installation is needed.

| Method string | Description |
|---|---|
| `doppler`     | Doppler reconnaissance defense (TNSM 2024) |
| `onset_v3`    | ONSET DDoS defense — edge-flow formulation (TDSC 2025) |
| `onset_v2`    | ONSET DDoS defense — path-flow formulation (TDSC 2025) |
| `onset`       | Original topology programming formulation (OptSys 2021) |
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

The expected smoke result: MLU `0.804324`, loss `0.0`, throughput `1.0`.

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
│   └── engine.py           # ECMP and MCF traffic engineering
└── utilities/              # Graph I/O, plotting, logging, path utilities
```

## License

MIT
