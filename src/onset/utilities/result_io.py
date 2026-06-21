"""Small readers for the simulator's legacy scalar result files."""

from __future__ import annotations


def read_result_val(result_file: str) -> float:
    with open(result_file, encoding="utf-8") as result_stream:
        result_stream.readline()
        fields = result_stream.readline().split()
    if len(fields) < 3:
        raise ValueError(f"Malformed scalar result file: {result_file}")
    return 0.0 if fields[2] == "-nan" else float(fields[2])
