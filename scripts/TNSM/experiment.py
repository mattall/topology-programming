import sys
import glob
import os
sys.path.insert(0, "src/")
sys.path.insert(0, "scripts/")
from experiment_params import *
from copy import deepcopy

DEBUG = False

def experiment(args):
    """Run one iteration of a topology programming experiment

    Args:
        args (tuple[network, t_class, scale, te_method, tp_method]):    
    Expects to find a traffic matrix at the following path.
        traffic_file = f"data/traffic/{t_class}_{network}-tm"

    Scales the traffic matrix by `scale`.

    te_method is the traffic engineering scheme used to forward traffic.

    tp_method is the topology programming method used to adapt the topology
    in light of traffic needs.

    Returns:
        str or None: Path to location of experiment report.
    """
    from onset.utilities.logger import logger
    from onset.utilities.post_process import read_result_val
    from onset.simulator import Simulation

    pid = os.getpid()
    logger.info(f"Process-{pid} started with data: {args}")
    network, t_class, scale, te_method, tp_method = args
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
        if DEBUG:
            pass
        else:
            result = my_sim.perform_sim(
                demand_factor=demand_factor, repeat=repeat[tp_method]
            )


    report_path = f"data/reports/{experiment_name}.csv"
    if DEBUG:
        pass
    elif isinstance(result, dict):
        # save this result data to a new var, calling run_sim will nuke result.
        result = deepcopy(result)
        # make sure we have now found the correct result path. 
        curr_res_path = get_result_path(my_sim)
        if curr_res_path == res_path:
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
        else:
            logger.error(f"After sim, result path does not match. Expected: {res_path} got: {curr_res_path}")
    else:
        logger.error("Did not have a result to write. Execution of sim failed.")

if __name__ == "__main__":
    network = "Tinet"
    traffic = "background"
    scale = "1.4"
    te = "semimcfraekeft"
    tp = "greylambda"
    experiment((network,traffic,scale,te,tp))

    # args = ("Comcast","background","0.8","mcf","greylambda")
    # args = ("Comcast","background-plus-flashcrowd","0.3","mcf","greylambda")
    # experiment(*args)
