#!/usr/bin/env python3
"""Run a small nonzero ANS/ECMP integration test through YATES."""

from onset.simulator import Simulation


def main() -> None:
    result = Simulation(
        "ANS",
        18,
        "smoke_ans_yates_ecmp",
        iterations=1,
        te_method="-ecmp",
        traffic_file="data/traffic/ANS_coremelt_every_link_2.00e+11.txt",
        topology_programming_method="baseline",
    ).perform_sim()
    congestion = result["Congestion"][0]
    loss = result["Loss"][0]
    throughput = result["Throughput"][0]
    assert congestion > 0
    assert loss == 0
    assert throughput == 1
    print(f"ANS ECMP smoke passed: MLU={congestion}, loss={loss}, throughput={throughput}")


if __name__ == "__main__":
    main()
