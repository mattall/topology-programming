import ipaddress
import os

CLEAN_START = True
SEED = 0
PLOT_DIR = "data/plots/"

IPV4 = ipaddress.IPv4Address._ALL_ONES
IPV4LENGTH = 32
SCRIPT_HOME = os.path.abspath(".")
USER_HOME = os.path.join(os.path.expanduser("~")) # For Yates executable
ZERO_INDEXED = False
PLT_BASE = 3
PLT_HIGHT = 1.854