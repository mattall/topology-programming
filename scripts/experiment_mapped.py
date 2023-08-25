# experiment.py
import sys
sys.path.insert(0, "/home/m/src/topology-programming/src/")
from onset.simulator import Simulation
from experiment_params import *

def experiment_mapped(args):
    te_method, tp_method, network, t_class, scale = args
    experiment_name = "-".join((te_method, tp_method, network, t_class, scale))
    traffic_file = f"data/traffic/{t_class}_{network}-tm"
    my_sim = Simulation(
        network,
        hosts[network],
        experiment_name,
        iterations=1,
        fallow_transponders=n_ftx[tp_method],
        te_method="-" + te_method,
        traffic_file=traffic_file,
        fallow_tx_allocation_strategy="static",
        topology_programming_method=tp_method,
        congestion_threshold_upper_bound=0.99999,
        congestion_threshold_lower_bound=0.99999,
    )
    demand_factor = float(scale) * mcf_loss_factor[network][t_class]
    result = my_sim.perform_sim(
        demand_factor=demand_factor, repeat=repeat[tp_method]
    )
    report_path = f"data/reports/{experiment_name}.csv"
    with open(report_path, "w") as report_fob:
        [report_fob.write(f"{key};") for key in result.keys()]
        report_fob.write("\n")
        [report_fob.write(f"{result[key][-1]};") for key in result.keys()]
        report_fob.write("\n")
    return f"data/reports/{experiment_name}.csv"


def experiment(te_method, tp_method, network, t_class, scale):
    experiment_name = "-".join((te_method, tp_method, network, t_class, scale))
    traffic_file = f"data/traffic/{t_class}_{network}-tm"
    my_sim = Simulation(
        network,
        hosts[network],
        experiment_name,
        iterations=1,
        fallow_transponders=n_ftx[tp_method],
        te_method="-" + te_method,
        traffic_file=traffic_file,
        fallow_tx_allocation_strategy="static",
        topology_programming_method=tp_method,
        congestion_threshold_upper_bound=0.99999,
        congestion_threshold_lower_bound=0.99999,
    )
    demand_factor = float(scale) * mcf_loss_factor[network][t_class]
    result = my_sim.perform_sim(
        demand_factor=demand_factor, repeat=repeat[tp_method]
    )
    report_path = f"data/reports/{experiment_name}.csv"
    with open(report_path, "w") as report_fob:
        [report_fob.write(f"{key};") for key in result.keys()]
        report_fob.write("\n")
        [report_fob.write(f"{result[key][-1]};") for key in result.keys()]
        report_fob.write("\n")
    return f"data/reports/{experiment_name}.csv"
