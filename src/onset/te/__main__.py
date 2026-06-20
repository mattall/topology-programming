"""Command-line entry point for one internal TE evaluation."""

import argparse

from onset.te import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topology", help="DOT topology file")
    parser.add_argument("traffic", help="flattened traffic-matrix file")
    parser.add_argument("hosts", help="ordered host-list file")
    parser.add_argument("--method", choices=("ecmp", "mcf"), required=True)
    parser.add_argument(
        "--out", required=True, help="result directory or data/results-relative path"
    )
    parser.add_argument(
        "--budget", type=int, default=3, help="maximum ECMP paths per pair"
    )
    args = parser.parse_args()

    result = evaluate(
        args.topology,
        args.traffic,
        args.hosts,
        args.method,
        args.out,
        budget=args.budget,
    )
    print(
        f"{args.method}: MLU={result.max_congestion:.6f}, "
        f"loss={result.congestion_loss:.6f}, throughput={result.throughput:.6f}"
    )


if __name__ == "__main__":
    main()
