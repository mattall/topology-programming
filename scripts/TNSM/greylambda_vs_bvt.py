#################################################################################################
#              | Scenario 1 (SNR degradation)                |   Scenario 2 (Fiber Cut)         #
#***********************************************************************************************#
# Case 1. TE   |                                             |                                  #
#              | Topology: (missing core link).              |   Topology: (missing core link). #
#################################################################################################
# Case 2. TE w/ BVT                                          |                                  #
#              | Topology: (partial capacity on core link).  |   Topology: (missing core link). #
#################################################################################################
# Case 3. TE w/ GreyLambda                                   |                                  #
#              | Topology: (missing core link).              |   Topology: (missing core link). #
#################################################################################################
# Case 4. TE w/ Greylambda & BVT                             |                                  #
#              | Topology: (partial capacity on core link).  |   Topology: (missing core link). #
#################################################################################################›

# Scenario 1. Partial link failure (loss of SNR)
# Scenario 2. Total Link Failure (loss of light)

# X - demand scale.
# Y - Throughput.

import os
import sys
from copy import deepcopy
import multiprocessing
from datetime import datetime

sys.path.insert(0, "src/")
sys.path.insert(0, "scripts/")
DEBUG = False

def experiment(args):
    # args expected:
    #  (network, t_class, scale, te_method, tp_method)
    from onset.utilities.logger import logger
    from onset.utilities.post_process import read_result_val
    from onset.simulator import Simulation

    hosts = {
        "Comcast": 149,
        "Comcast_63_133": 149,
        "Tinet": 53
        # "Verizon": 116,
        # "azure": 113,
        # "Zayo": 96,
        # "b4": 54,
    }

    mcf_loss_factor = {
        "Comcast": {
            "background": 1.210373,
            "background-plus-flashcrowd": 1.633346,
        },
        "Comcast_63_133": {
            "background": 1.210373,
            "background-plus-flashcrowd": 1.633346,
        },

        "Tinet": {"background": 297.500000},
    }
    repeat = {"TBE": False, "greylambda": True, "TE": False}
    n_ftx = {"TE": 0, "TBE": 0, "greylambda": 2}

    pid = os.getpid()
    logger.info(f"Process-{pid} started with data: {args}")

    # te_method, tp_method, network, t_class, scale = args
    network, t_class, scale, te_method, tp_method, line_code = args
    # if "63_133" in network:
    #     experiment_name = "-".join([str(scale), te_method, tp_method, line_code])
    # else:
        
    experiment_name = "-".join([str(scale), tp_method, line_code])

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
        line_code=line_code
    )
    demand_factor = float(scale) * mcf_loss_factor[network][t_class]
    tracked_vars = ["Congestion", "Loss", "Throughput"]
    file_of_var = {
        "Congestion": "MaxExpCongestionVsIterations.dat",
        "Loss": "CongestionLossVsIterations.dat",
        "Throughput": "TotalThroughputVsIterations.dat",
    }
    result = {}
    try:
        res_path =  my_sim.perform_sim(
            repeat=repeat[tp_method], self.demand_factor=demand_factor, dry=True
        )
        if os.path.exists(res_path):
            for tv in tracked_vars:
                result[tv] = [read_result_val(os.path.join(res_path, file_of_var[tv]))]
            logger.info(f"Read Congestion, Loss, Throughput values from: {res_path}")
        else: 
            raise FileNotFoundError
        
    except FileNotFoundError:
        logger.error(
            f"Failed to read Congestion, Loss, Throughput values. Expected them in: {res_path}"
        )
        if DEBUG:
            pass
        else:
            result = my_sim.perform_sim(
                self.demand_factor=demand_factor, repeat=repeat[tp_method]
            )

    report_path = f"data/reports/{experiment_name}.csv"
    if DEBUG:
        pass
    elif isinstance(result, dict):
        # save this result data to a new var, calling run_sim will nuke result.
        result = deepcopy(result)
        # make sure we have now found the correct result path.
        curr_res_path =  my_sim.perform_sim(
            repeat=repeat[tp_method], self.demand_factor=demand_factor, dry=True
        )
        if curr_res_path == res_path:
            buffer = f"{network},{t_class},{scale},{te_method},{tp_method},{line_code},"
            # report_fob.write(
            #     f"{network},{t_class},{scale},{te_method},{tp_method},{line_code},"
            # )
            writable = True
            for tv in tracked_vars:
                if (
                    tv in result
                    and isinstance(result[tv], list)
                    and len(result[tv]) > 0
                ):
                    if tv == tracked_vars[-1]:
                        buffer += f"{result[tv][-1]}\n"
                        # report_fob.write(f"{result[tv][-1]}\n")
                    else:
                        buffer += f"{result[tv][-1]},"
                        # report_fob.write(f"{result[tv][-1]},")
                else:
                    writable = False
            if writable:
                with open(report_path, "w") as report_fob:
                    report_fob.write(buffer)
                logger.info(f"Wrote report to {report_path}")
            else:
                logger.error(f"Couldn't write result: {result}")

            return report_path
        else:
            logger.error(
                f"After sim, result path does not match. Expected: {res_path} got: {curr_res_path}"
            )
    else:
        logger.error("Did not have a result to write. Execution of sim failed.")

def initialize_params():
    traffic_file = "background"
    topo_whole = "Comcast"
    topo_missing_link = "Comcast_63_133"

    scenarios = ["Partial Failure", "Total Failure"]
    cases = ["TE", "TE+BVT", "TE+GreyLambda", "TE+BVT+GreyLambda"]

    from collections import defaultdict

    experiments = defaultdict(lambda: defaultdict())
    #  (network, t_class, scale, te_method, tp_method)

    experiments["Total Failure"]["TE"] = (
        topo_missing_link,
        traffic_file,
        "mcf",
        None,
        None,
    )
    experiments["Total Failure"]["TE+BVT"] = (
        topo_missing_link,
        traffic_file,
        "mcf",
        None,
        None,
    )
    experiments["Total Failure"]["TE+GreyLambda"] = (
        topo_missing_link,
        traffic_file,
        "mcf",
        "greylambda",
        None,
    )
    experiments["Total Failure"]["TE+BVT+GreyLambda"] = (
        topo_missing_link,
        traffic_file,
        "mcf",
        "greylambda",
        None,
    )

    experiments["Partial Failure"]["TE"] = (
        topo_missing_link,
        traffic_file,
        "mcf",
        None,
        None,
    )
    experiments["Partial Failure"]["TE+BVT"] = (
        topo_whole,
        traffic_file,
        "mcf",
        None,
        "BVT",
    )
    experiments["Partial Failure"]["TE+GreyLambda"] = (
        topo_missing_link,
        "mcf",
        "greylambda",
        None,
    )
    experiments["Partial Failure"]["TE+BVT+GreyLambda"] = (
        topo_whole,
        traffic_file,
        "mcf",
        "greylambda",
        "BVT",
    )


    S = set(
        [
            (topo_missing_link, traffic_file, "mcf", "TE",          "fixed"),        # fiber cut with TE
            (topo_missing_link, traffic_file, "mcf", "TE",          "fixed"),        # fiber cut with BVT
            (topo_missing_link, traffic_file, "mcf", "greylambda",  "fixed"),        # fiber cut with GL
            (topo_missing_link, traffic_file, "mcf", "greylambda",  "fixed"),        # fiber cut with BVT+GL
            (topo_missing_link, traffic_file, "mcf", "TE",          "fixed"),        # SNR failure with TE
            (topo_whole,        traffic_file, "mcf", "TE",          "BVT"),          # SNR failure with BVT
            (topo_missing_link, traffic_file, "mcf", "greylambda",  "fixed"),        # SNR failure with GL
            (topo_whole,        traffic_file, "mcf", "greylambda",  "BVT"),          # SNR failure with BVT+GL
        ]
    )

    params = []
    for a, b, c, d, e in S:
        for i in range(10, 21):
            params.append((a, b, i / 10.0, c, d, e))
    
    return params


def error_callback(error):
    print(f"Process {os.getpid()} compromised. Got an Error: {error}\n", flush=True)

def expected_callback(result_file):
    print(f"Process {os.getpid()} complete. Wrote result to: {result_file}\n", flush=True)


if __name__ == "__main__":
    experiment_params = initialize_params()
    # if len(sys.argv) > 1 and sys.argv[1] == "1":
    #     experiment_params = experiment_params[:30]
    # else:
    #     experiment_params = experiment_params[30:]
    # for e in experiment_params:
    #     experiment(e)
    # newp = []
    # for p in experiment_params:
    #     if "63" in p[0]:
    #         newp.append(p)
    # for p in newp:
    #     print(p)
    # print(len(newp))
    ep = [e for e in experiment_params if "BVT" in e]
    for e in ep:
        print(e)
    print(len(ep))
    experiment_params = ep

    PARALLEL = True
    # PARALLEL = False

    if PARALLEL: 
        pool = multiprocessing.Pool()
        pool.map_async(
            experiment, 
            experiment_params, 
            callback=expected_callback, 
            error_callback=error_callback)
        pool.close()
        pool.join()
    else:
        for ep in experiment_params:
            experiment(ep)

