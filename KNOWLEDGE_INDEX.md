# Topology Programming Knowledge Index

This is the fast path for getting oriented in this repository. It points to
the project map in `docs/knowledge/` and calls out the most important files to
read first.

## Start Here

1. Read `docs/knowledge/README.md` for the project shape, vocabulary, and
   recommended reading order.
2. Read `docs/knowledge/recent-direction.md` to understand where the project
   seemed to be going in the most recent commits.
3. Read `docs/knowledge/architecture.md` for the simulator runtime and core
   package responsibilities.
4. Read `docs/knowledge/experiments.md` before touching Doppler, TDSC, TNSM, or
   batch scripts.
5. Read `docs/knowledge/data-and-environment.md` before trying to run anything.

## High-Signal Files

- `src/onset/simulator.py`: main orchestration layer for simulations.
- `src/onset/optimization_two.py`: current optimization workhorse, especially
  for Doppler-era runs.
- `scripts/Doppler/exp-v2.py`: latest Doppler experiment driver and batch
  command generator.
- `scripts/Doppler/experiment_params.py`: network, host-count, method, repeat,
  and demand-scale constants for Doppler experiments.
- `scripts/Doppler/batch/CEN.batch`: example SLURM array workflow.
- `scripts/setup-yates.sh`: reproducible YATES/Frenetic setup in an isolated
  opam switch.
- `scripts/check-env.sh`: local preflight for Python, YATES, optional Gurobi,
  and core Python imports.
- `scripts/smoke_ans_ecmp.py`: nonzero ANS/ECMP end-to-end smoke test.
- `external/yates`: pinned git submodule for Cornell Netlab YATES.
- `scripts/TDSC/`: crossfire/rolling attack evaluation scripts and plotting.
- `src/onset/network_model.py`: graph import and network model enrichment.
- `src/onset/utilities/`: graph, traffic, result, plotting, and Yates helpers.

## Current Read

The recent work appears to have moved from broad simulator development toward
running parameterized experiment campaigns. The freshest Doppler path generates
explicit commands for scheduler execution, evaluates `top_k`, fallow
transponder counts, candidate-link choice methods, and optimizer time limits,
then writes compact CSV reports in `data/reports/`.
