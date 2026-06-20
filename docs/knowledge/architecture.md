# Architecture

The codebase is organized around a simulator that prepares inputs, invokes
topology programming methods, evaluates resulting topologies, and writes
metrics. Most experiment scripts are thin orchestration around that simulator.

## Package Map

- `src/onset/simulator.py`
  - Main `Simulation` class.
  - Owns experiment IDs, result directories, traffic matrix selection/scaling,
    topology export, topology programming dispatch, Yates invocation, and metric
    collection.
- `src/onset/optimization_two.py`
  - Active `Link_optimization` implementation.
  - Uses Gurobi models for Doppler/onset variants and related optimization
    methods.
  - Builds candidate links, path/tunnel sets, graph similarity terms, max-link
    utilization terms, and solution outputs.
- `src/onset/doppler.py`
  - Older or narrower Doppler optimization implementation. It shares concepts
    with `optimization_two.py`, but recent simulator imports point to
    `optimization_two.Link_optimization`.
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
    plotting, post-processing, logging, and Yates wrappers.
- `src/onset/utilities/executables.py`
  - Resolves external command dependencies such as YATES from environment
    variables or `PATH`.

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
6. Run traffic engineering/evaluation, usually through YATES helpers. The
   executable is resolved from `YATES_BIN` or `yates` on `PATH`.
7. Write result files such as:
   - `MaxExpCongestionVsIterations.dat`
   - `CongestionLossVsIterations.dat`
   - `TotalThroughputVsIterations.dat`
   - Doppler-specific files like `TotalSolutions.dat`, `OptTime.dat`,
     `DopplerMinMLU.dat`, `CurrTopoID.dat`, and `OptimalTopoID.dat`.

## Traffic Engineering Boundary

YATES is currently both a routing engine and a statistics engine. For each
evaluated topology, the simulator passes topology, actual/predicted traffic,
hosts, and `te_method` to YATES. YATES writes path assignments and metrics such
as MLU, loss, throughput, and path counts back under `data/results/`.

The experiment scripts primarily exercise `-ecmp` and `-mcf`. Some historical
paths also name `-semimcfraeke`, `-semimcfraekeft`, and `-semimcfecmp`, but the
recent Doppler/TDSC grids center on ECMP and MCF. In `onset_v3`, ECMP candidate
topologies may additionally be ranked by invoking YATES on each candidate.

This boundary is the target for a future in-core TE replacement: preserve the
result-file/metric contract while implementing ECMP and MCF routing directly.

## Doppler Path

The current Doppler path is spread across:

- `scripts/Doppler/exp-v2.py`: creates one `Simulation` per parameter tuple.
- `src/onset/simulator.py`: detects `Doppler` and invokes `doppler_method`.
- `src/onset/optimization_two.py`: builds and solves the Gurobi optimization.

Important Doppler parameters:

- `top_k`: how many optimizer solutions or topology alternatives to consider.
- `fallow_transponders`: optical resources available per node or under an
  allocation policy.
- `candidate_link_choice_method`: candidate set mode, commonly `max` or
  `conservative`.
- `optimizer_time_limit_minutes`: Gurobi time limit for optimization calls.
- `scale_down_factor`: numerical scaling for demand values in optimization.

## Candidate Links and Paths

`optimization_two.Link_optimization.initialize_candidate_links()` supports
several candidate sets:

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
