from onset.simulator import Simulation
from itertools import product
from os import path
from sys import exit

starting_demand_factor = 1

te_methods = ["mcf", "semimcfraekeft", "ncflow"]
tp_methods = ["TE", "TBE", "greylambda"]
networks = ["Comcast", "Verizon", "Zayo", "azure", "b4"]
t_classes = ["background", "background-plus-flashcrowd"]

TE_name = {"-mcf": "MCF", "-semimcfraekeft": "SMORE", "-ncflow": "NCFlow"}
n_ftx = {"TE": 0, "TBE": 0, "greylambda": 2}
hosts = {"Comcast": 149, "Verizon": 116, "azure": 113, "Zayo": 96, "b4": 54}

experiment_combinations = product(te_methods, tp_methods, networks, t_classes)

# for traffic, experiment_name, ftx, strat in params:
for te_method, tp_method, network, t_class in experiment_combinations:
    experiment_name = "-".join((te_method, tp_method, network, t_class))
    traffic_file = f"data/traffic/{t_class}_{network}-tm",
    demand_factor = starting_demand_factor
    my_sim = Simulation(
        network,
        hosts,
        experiment_name,
        iterations=1,
        fallow_transponders=n_ftx[tp_method],
        te_method= "-" + te_method,
        traffic_file=traffic_file,
        fallow_tx_allocation_strategy="static", 
        topology_programming_method=tp_method,
        congestion_threshold_upper_bound=0.99999,
        congestion_threshold_lower_bound=0.99999,
    )  

    try:
        report_path = (
            f"data/reports/{experiment_name}.csv"
        )
        if path.exists(report_path):
            print_header = False
            report_fob = open(report_path, "a")
        else:
            print_header = True
            report_fob = open(report_path, "w")

        prev_path = ""
        while True:
            result = my_sim.perform_sim(
                unit="Mbps", demand_factor=demand_factor, repeat=True
            )

            if print_header:
                [report_fob.write(f"{key};") for key in result.keys()]
                report_fob.write(f"Demand Factor\n")
                print_header = False

            [report_fob.write(f"{result[key][-1]};") for key in result.keys()]
            report_fob.write(f"{demand_factor}\n")

            if result["Loss"][-1] > 0.4:
                break
            else:
                demand_factor *= 1.1

    except KeyboardInterrupt:
        report_fob.close()
        exit()

    except Exception as e:
        print(e)

    finally:
        report_fob.close()
