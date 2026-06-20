# Knowledge Base

This repository is a Python research codebase for topology programming in
reconfigurable optical networks. It combines a simulator, optimization models,
traffic engineering evaluation, attack/defense experiments, and plotting/report
scripts used across several experiment lines.

The best mental model is:

- `src/onset/` is the reusable package.
- `src/onset/simulator.py` is the central coordinator.
- `src/onset/optimization_two.py` is the active optimization workhorse.
- `scripts/` contains publication- or campaign-specific experiment workflows.
- `data/` contains topologies, traffic matrices, host files, paths, reports,
  results, and other generated or input artifacts.

## Reading Order

1. `recent-direction.md`: what the latest commits suggest was happening.
2. `architecture.md`: core package layout and simulator lifecycle.
3. `experiments.md`: Doppler, TDSC, TNSM, and older experiment workflows.
4. `data-and-environment.md`: inputs, outputs, dependencies, and run hazards.

## Core Vocabulary

- Topology programming: modifying logical topology by adding, dropping, or
  reassigning links/circuits/transponders.
- TE: traffic engineering method passed through to evaluation, often `-ecmp`
  or `-mcf`.
- TP: topology programming method, such as `Doppler`, `greylambda`, `TBE`,
  `BVT`, `onset`, `onset_v2`, or `onset_v3`.
- Fallow transponders: spare optical resources available for reconfiguration.
- Candidate links: possible links/circuits the optimizer may add or choose
  among.
- MLU: maximum link utilization, a central result metric.
- YATES: external traffic engineering/evaluation tool invoked by helper code.
  Use `YATES_BIN` or put `yates` on `PATH`.

## What Is Polished vs. Researchy

Treat this as a living research repo. Some paths are clearly current and some
are legacy. The package is installable in the PyScaffold style, but actual
experiment execution depends on external data and YATES. Gurobi is additionally
required for MCF and optimization-backed topology methods. Use
`scripts/check-env.sh` for a quick local dependency check and
`scripts/smoke_ans_ecmp.py` for a known end-to-end ECMP run.
