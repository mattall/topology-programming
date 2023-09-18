import sys
import glob
import os
import logging
sys.path.insert(0, "/home/mhall7/durairajanlab.matt/topology-programming/src/")
from onset.simulator import Simulation
from onset.utilities.post_process import read_result_val
sys.path.insert(0, "/home/mhall7/durairajanlab.matt/topology-programming/scripts/")
from experiment_params import *
# from onset.utilities.logger import logger


def experiment_mapped(args):
    from onset.utilities.logger import logger
    pid = os.getpid()
    logger.info(f"Process-{pid} started with data: {args}")
    
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
    tracked_vars = ["Congestion", "Loss", "Throughput"]
    file_of_var = { "Congestion": "MaxExpCongestionVsIterations.dat",
                    "Loss":"CongestionLossVsIterations.dat",
                    "Throughput": "TotalThroughputVsIterations.dat",
                   }
    result = {}
    try:
        def get_result_path(sim):
            #helper function 
            result_path = my_sim.perform_sim(
                repeat=repeat[tp_method], demand_factor=demand_factor, dry=True
            )
            # result_path = my_sim.perform_sim(repeat=repeat[tp_method], demand_factor=demand_factor)

            return result_path

        res_path = get_result_path(my_sim)
        for tv in tracked_vars:
            result[tv] = [
                read_result_val( os.path.join(res_path, file_of_var[tv]) )
                ]
        logger.info(f"Read Congestion, Loss, Throughput values from: {res_path}")

    except FileNotFoundError:
        logger.error(f"Failed to read Congestion, Loss, Throughput values. Expected them in: {res_path}")
        result = my_sim.perform_sim(
            demand_factor=demand_factor, repeat=repeat[tp_method]
        )


    report_path = f"data/reports/{experiment_name}.csv"
    with open(report_path, "w") as report_fob:
        report_fob.write(f"{network},{t_class},{scale},{te_method},{tp_method},")
        for tv in tracked_vars:
            if tv in result \
            and isinstance(result[tv], list) \
            and len(result[tv]) > 0:
                if tv == tracked_vars[-1]:
                    report_fob.write(f"{result[tv][-1]}\n")
                else:
                    report_fob.write(f"{result[tv][-1]},")
    logger.info(f"Wrote report to {report_path}")
    return report_path

def experiment(te_method, tp_method, network, t_class, scale):
    args = (te_method, tp_method, network, t_class, scale)
    experiment_mapped(args)
    # experiment_name = "-".join((te_method, tp_method, network, t_class, scale))
    # if was_completed(experiment_name):
    #     return f"data/reports/{experiment_name}.csv"
    # traffic_file = f"data/traffic/{t_class}_{network}-tm"
    # my_sim = Simulation(
    #     network,
    #     hosts[network],
    #     experiment_name,
    #     iterations=1,
    #     fallow_transponders=n_ftx[tp_method],
    #     te_method="-" + te_method,
    #     traffic_file=traffic_file,
    #     fallow_tx_allocation_strategy="static",
    #     topology_programming_method=tp_method,
    #     congestion_threshold_upper_bound=0.99999,
    #     congestion_threshold_lower_bound=0.99999,
    # )
    # demand_factor = float(scale) * mcf_loss_factor[network][t_class]
    # result = my_sim.perform_sim(
    #     demand_factor=demand_factor, repeat=repeat[tp_method]
    # )
    # report_path = f"data/reports/{experiment_name}.csv"
    # with open(report_path, "w") as report_fob:
    #     [report_fob.write(f"{key};") for key in result.keys()]
    #     report_fob.write("\n")
    #     try:
    #         [report_fob.write(f"{result[key][-1]};") for key in result.keys()]
    #     except:
    #         report_fob.write(f"ERROR: result format unexpected. {result}")
    #     report_fob.write("\n")
    # return f"data/reports/{experiment_name}.csv"

def was_completed(experiment_name):
    top_dir = f"data/results/*{experiment_name}*"
    # Recursively find all directories matching `experiment_name`
    experiment_directories = [f for f in glob.glob(os.path.join(f"{top_dir}", '**'), recursive=True) if os.path.isdir(f)]
    
    for experiment_dir in experiment_directories:
        if not os.path.isdir(experiment_dir): continue
        dat_files = glob.glob(os.path.join(experiment_dir, '**', '*.dat'), recursive=True)
        if dat_files:
            return True
        else:
            return False

def was_not_completed(experiment_name):
    return not was_completed(experiment_name)

if __name__ == "__main__":
    # a = 'mcf'
    # b = "greylambda"
    # c = "Comcast"
    # d = "background"
    # e = "1.1"


    # a = "semimcfraekeft"
    # b = "TBE"
    # c = "Comcast"
    # d = "background"
    # e = "1.0"

    # a = "semimcfraekeft"
    # b = "TBE"
    # c = "Tinet"
    # d = "background"
    # e = "1.0"

    a = "semimcfraekeft"
    b = "greylambda"
    c = "Tinet"
    d = "background"
    e = "1.2"
    experiment(a,b,c,d,e)

    args = ("Comcast","background","0.8","mcf","greylambda")
    args = ("Comcast","background-plus-flashcrowd","0.3","mcf","greylambda")
    experiment(*args)
