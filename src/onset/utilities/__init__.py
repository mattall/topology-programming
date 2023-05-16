from os.path import realpath, curdir
import os
import matplotlib.pyplot as plt


SCRIPT_HOME = os.path.join(os.path.expanduser("~"), "topology_programming")
USER_HOME = os.path.join(os.path.expanduser("~"))
ZERO_INDEXED = False


plt.rcParams.update(
    {
        "figure.constrained_layout.use": True,
        "font.size": 26,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 2,
        "xtick.major.size": 12,
        "ytick.major.size": 12,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.minor.size": 6,
        "ytick.minor.size": 6,
        "xtick.minor.width": 2,
        "ytick.minor.width": 2,
        "lines.linewidth": 2,
    }
)
PLT_BASE = 3
PLT_HIGHT = 1.854
