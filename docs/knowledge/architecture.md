# Architecture

The codebase is organized around a simulator that prepares inputs, invokes
topology programming methods, evaluates resulting topologies, and writes
metrics. Most experiment scripts are thin orchestration around that simulator.

## Package Map

- `src/onset/simulator.py`
  - Main `Simulation` class.
  - Owns experiment IDs, result directories, traffic matrix selection/scaling,
    topology export, topology programming dispatch, TE evaluation, and metric
    collection.
- `src/onset/open_doppler.py`
  - Active HiGHS-based MILP solver supporting all four topology programming
    formulations (doppler, onset, onset_v2, onset_v3).
  - Builds and solves edge-flow and path-flow optimization models.
- `src/onset/preprocessing.py`
  - Builds the `OptimizationProblem` dataclass: candidate links, tunnels,
    demand loading, and fallow-transponder logic extracted from the legacy
    Gurobi code.
- `src/onset/gurobi_doppler.py`
  - Legacy Gurobi Doppler implementation. Current dispatch goes through
    `method_registry.py` → `open_doppler.py` (HiGHS).
- `src/onset/network_model.py`
  - Imports `.gml` or `.json` topologies into a richer NetworkX model.
  - Adds router/client structure, interface metadata, node/link types, and
    default capacities.
- `src/onset/alpwolf.py`
  - Topology/optical-network manipulation layer used by the simulator.
- `src/onset/attacker.py`
  - Traffic-matrix generation for attack scenarios such as coremelt,
    crossfire, flash-crowd, and rolling attacks.
- `src/onset/utilities/`
  - Graph conversion/path helpers, traffic-matrix generators, flow utilities,
    plotting, post-processing, and logging.
- `src/onset/te/`
  - In-process ECMP routing, MCF optimization through SciPy/HiGHS, congestion
    simulation, and result-file generation.

## Simulator Lifecycle

At a high level, `Simulation.perform_sim(...)` does this:

1. Build an experiment ID from network, test name, TE method, TP method,
   fallow transponders, candidate-link method, thresholds, scale factors, and
   Doppler parameters.
2. Create or locate result directories under `data/results/`.
3. Prepare the current traffic matrix, either from generated demand or an
   explicit `traffic_file`.
4. Export or locate the current topology and hosts files.
5. Dispatch the topology programming method:
   - `cli`
   - `cache`
   - `onset`
   - `onset_v2`
   - `onset_v3`
   - `OTP`
   - `greylambda`
   - `BVT`
   - `TBE`
   - any method containing `Doppler`
6. Run ECMP or MCF traffic engineering through `onset.te.evaluate`.
7. Write result files such as:
   - `MaxExpCongestionVsIterations.dat`
   - `CongestionLossVsIterations.dat`
   - `TotalThroughputVsIterations.dat`
   - Doppler-specific files like `TotalSolutions.dat`, `OptTime.dat`,
     `DopplerMinMLU.dat`, `CurrTopoID.dat`, and `OptimalTopoID.dat`.

## Traffic Engineering Boundary

`onset.te.evaluate` is both the routing and statistics boundary. For each
evaluated topology, the simulator passes topology, traffic, hosts, `te_method`,
and result path. The engine writes path assignments and metrics such as MLU,
loss, throughput, and path counts under `data/results/`.

The experiment scripts primarily exercise `-ecmp` and `-mcf`. Some historical
paths name `-semimcfraeke`, `-semimcfraekeft`, and `-semimcfecmp`; these legacy
methods are not implemented by the internal engine and now fail with a clear
error. In `onset_v3`, ECMP candidate topologies are ranked in parallel through
the same internal evaluator.

Compatibility details are deliberate: capacity suffixes use YATES's binary
multipliers, non-switch access links receive its historical 100x multiplier,
and ECMP is capped deterministically by the path budget (currently three).

## Doppler Path

The current Doppler path is spread across:

- `scripts/Doppler/exp-v2.py`: creates one `Simulation` per parameter tuple.
- `src/onset/simulator.py`: detects `Doppler` and dispatches through `method_registry.py`.
- `src/onset/open_doppler.py`: builds and solves the HiGHS optimization.

Important Doppler parameters:

- `top_k`: how many optimizer solutions or topology alternatives to consider.
- `fallow_transponders`: optical resources available per node or under an
  allocation policy.
- `candidate_link_choice_method`: candidate set mode, commonly `max` or
  `conservative`.
- `optimizer_time_limit_minutes`: time limit for optimization calls.
- `scale_down_factor`: numerical scaling for demand values in optimization.

## Candidate Links and Paths

Candidate link and tunnel logic lives in `src/onset/preprocessing.py`, which
builds the `OptimizationProblem` dataclass. Supported candidate sets:

- `max`: broad candidate set, including reachable non-existing node pairs
  within distance constraints.
- `liberal`: candidate links derived from neighborhoods around top central
  edges.
- `conservative`: smaller candidate set around top central edges.

Path/tunnel data may be cached under `data/paths/optimization/`, often as
pickle files in newer code.

## Result Flow

Experiment scripts usually collect the simulator's per-run `.dat` files into
CSV reports under `data/reports/`. Plotting scripts then consume those reports
or raw result directories.
