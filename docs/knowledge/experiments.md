# Experiments

Most practical work in this repo happens through scripts under `scripts/`.
They encode experiment grids, batch-run mechanics, report collection, and
plotting for different research threads.

## Doppler

Primary files:

- `scripts/Doppler/exp-v2.py`
- `scripts/Doppler/experiment.py`
- `scripts/Doppler/experiment_params.py`
- `scripts/Doppler/batch/CEN.batch`
- `scripts/Doppler/doppler.batch`

`exp-v2.py` is the most recent Doppler driver. In its current shape:

1. `main()` defines a parameter grid.
2. It prints one command line for each parameter tuple.
3. It exits.
4. When called with arguments, it runs one tuple by constructing a
   `Simulation`.

The expected traffic matrix path is:

```text
data/traffic/{traffic}_{network}-tm
```

The per-run report path is:

```text
data/reports/{te}-{tp}-{network}-{traffic}-{scale}-{n_ftx}-{top_k}-{candidate}-{time_limit}.csv
```

The report columns include network, traffic, scale, TE, TP, `top_k`, fallow
transponders, candidate-link choice, time limit, congestion, loss, throughput,
solution count, Doppler minimum MLU, optimization time, current topology ID,
and optimal topology ID.

### Batch Pattern

`scripts/Doppler/batch/CEN.batch` is a SLURM array example. It activates
`.venv`, reads one command from an args file based on `SLURM_ARRAY_TASK_ID`, and
runs it with Python.

A typical flow is:

```bash
python scripts/Doppler/exp-v2.py > scripts/Doppler/batch/CEN-args.txt
sbatch scripts/Doppler/batch/CEN.batch
```

Check and edit the array bounds in the batch file so they match the number of
commands in the args file.

## TDSC

Primary directory:

```text
scripts/TDSC/
```

Recent work added producer/consumer scripts and plotting for crossfire and
rolling attack evaluations:

- `crossfire_producer.py`
- `crossfire_consumer.py`
- `rolling_producer.py`
- `rolling_consumer.py`
- `plot_crossfire.py`
- `plot_rolling.py`
- `plot_link_change.py`
- `time_plotter.py`
- `num_paths_plt.py`

The TDSC scripts appear to focus on attack evaluation, link-change analysis,
and publication-style plots from existing simulator outputs.

There is also `scripts/TDSC/legacy/`, which preserves older experiment runners.
Treat these as historical reference unless a current script imports them.

## TNSM

Primary directory:

```text
scripts/TNSM/
```

This line compares TE, TBE, GreyLambda, BVT, and related approaches. Batch
scripts live under `scripts/TNSM/batch/`.

The simulator still contains explicit methods for:

- `greylambda`
- `BVT`
- `TBE`

So TNSM is not purely historical, but the most recent commit activity was more
Doppler/TDSC-focused.

## HotNets23

Primary directory:

```text
scripts/HotNets23/
```

This appears to be a smaller workflow around HotNets-era optimization and
traffic fetching. The simulator method `OTP` is labeled as PDP+OTP HotNets-23
in code comments.

## Command-Line Simulator

`src/onset/net_sim.py` is an older command-line wrapper around
`Simulation`. It accepts topology, host count, test name, TE method, strategy,
traffic file, fallow-transponder allocation, and post-processing options.

It is useful for understanding expected CLI concepts, but the freshest Doppler
campaign uses `scripts/Doppler/exp-v2.py` instead.

