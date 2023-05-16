from numpy import percentile


def diff_compare(diff_file, kind=None) -> float:
    if kind == "path":
        # path diff files have a new line '<' for paths removed, '>' for paths added.
        # Since paths are bi-directional, we divide the number of line differences by 2.
        with open(diff_file, "r") as fob:
            churn = 0
            for line in fob:
                if line.startswith("<"):
                    churn -= 1
                elif line.startswith(">"):
                    churn += 1
        return churn / 2

    else:
        v1 = 0
        v2 = 0
        v1_is_set = False
        v2_is_set = False
        with open(diff_file, "r") as fob:
            for line in fob:
                if line.startswith("<"):
                    assert (
                        not v1_is_set
                    ), "Error, expected diff file to have only one change!"
                    v1 = float(line.split()[3])
                    v1_is_set = True

                elif line.startswith(">"):
                    assert (
                        not v2_is_set
                    ), "Error, expected diff file to have only one change!"
                    v2 = float(line.split()[3])
                    v2_is_set = True
        if v1 == 0 and v2 == 0:
            percent_change = 0
        else:
            percent_change = 100 * (v2 - v1) / v1
        return percent_change
