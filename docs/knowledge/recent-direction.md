# Recent Direction

The latest commits suggest the project had moved from simulator mechanics into
experiment production, especially Doppler and TDSC evaluation.

## Commit Signals

- `3cf835d documentation and vendoring dependencies`
  - Added this knowledge tree, the YATES submodule, executable resolution, and
    environment checks.
  - Direction: make the research codebase recoverable by a new maintainer.
- `1887e44 Document and automate reproducible YATES setup`
  - Closed the historical external-engine setup loop and established ANS ECMP
    result parity before the internal replacement branch.

- `19b8f7f Update and rename README.rst to README.md`
  - Replaced the old README with a Markdown overview.
  - The new README is broad and partially aspirational; it does not capture the
    full current package layout or experiment workflow.
- `4655e04 Doppler experiments update`
  - Updated `scripts/Doppler/exp-v2.py`.
  - Updated `scripts/Doppler/experiment_params.py`.
  - Added/updated `scripts/Doppler/batch/CEN.batch`.
  - Changed `src/onset/simulator.py`, `src/onset/gurobi_optimization.py`, and
    `src/onset/utilities/graph.py`.
  - Direction: run Doppler experiments over explicit parameter tuples,
    including `top_k`, fallow transponder counts, candidate-link selection, and
    optimizer time limits.
- `9f32a19 TDSC eval scripts`
  - Added crossfire and rolling attack producers/consumers, plots, and report
    collectors under `scripts/TDSC/`.
  - Direction: evaluate attack scenarios and produce publication-style plots
    from simulator outputs.
- `6993a31 changed defaults`
  - Changed simulator defaults.
  - Direction: likely tuning experiment behavior rather than adding new
    architecture.

## Most Recent Doppler Workflow

`scripts/Doppler/exp-v2.py` is the strongest indicator of where work left off.
Its `main()` no longer directly runs the full product grid. Instead it prints
one command per parameter tuple and exits. Then, when invoked with arguments, it
runs exactly one experiment tuple.

That shape fits scheduler execution:

1. Generate command lines from the grid.
2. Save them into an args file.
3. Use a SLURM array script such as `scripts/Doppler/batch/CEN.batch`.
4. Each array job reads one line and executes `python <that command>`.

The current parameter axes in `exp-v2.py` include:

- network, recently `Gigapop` in the non-debug branch and `CEN` in debug.
- traffic class, usually `background`.
- demand scale.
- TE method, currently `mcf` in the recent Doppler runner.
- TP method, currently `Doppler`.
- `top_k`, generated as `0, 10, ..., 100`.
- number of fallow transponders.
- candidate-link choice method, especially `conservative` and `max`.
- optimizer time limit in minutes.

## Important Rough Edges

- `scripts/Doppler/experiment_params.py` has a likely typo:
  `hosts` contains `"Gigalpop": 10`, while recent `exp-v2.py` uses
  `"Gigapop"`. That would produce a key error unless fixed or overridden.
- `exp-v2.py` accepts `dry=False`, but still calls `perform_sim(...)` directly;
  the `dry` parameter is not currently used to skip work.
- `Makefile clean` removes generated outputs and `gurobi.log`. Use it
  carefully because it deletes `data/results/*`, `data/reports/*`, and `.temp/*`.
- Tests reference data files that may not exist in a fresh clone.
  Run `python -m pytest tests/ --ignore=tests/sim_import_json_test.py
  --ignore=tests/network_model_test.py --ignore=tests/generate_flows_test.py`
  to exercise the tests that do not depend on external data.
- ECMP, MCF, and SMORE (semi-oblivious MCF with Raecke decomposition) TE
  evaluation now live in `src/onset/te/`.
