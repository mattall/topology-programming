import sys
import glob
import os
import errno
sys.path.insert(0, "src/")
sys.path.insert(0, "scripts/")
from itertools import product
from experiment_params import *
from copy import deepcopy
import multiprocessing

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
    SCALE_DOWN_FACTOR = 10**7
    network, t_class, scale, te_method, tp_method, use_cached_result, top_k, n_ftx = args

    def get_result_path(sim):
        #helper function 
        result_path = my_sim.perform_sim(
            repeat=repeat[tp_method], demand_factor=demand_factor, dry=True
        )
        # result_path = my_sim.perform_sim(repeat=repeat[tp_method], demand_factor=demand_factor)

        return result_path


    def check_complete(my_sim, tracked_vars, result):    
        '''
        Checks for a completed experiment. populates result data if complete, or raises FileNotFound error if not. 
        '''
        res_path = get_result_path(my_sim)                
        for tv in tracked_vars:
            if RERUN_OK: 
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), res_path)
            result[tv] = [
                read_result_val( os.path.join(res_path, file_of_var[tv]) )
                ]

        logger.info(f"Read Congestion, Loss, Throughput values from: {res_path}")
        return result, res_path

    pid = os.getpid()
    logger.info(f"Process-{pid} started with data: {args}")
    
    experiment_name = "-".join((te_method, tp_method, network, t_class, scale))
    traffic_file = f"data/traffic/{t_class}_{network}-tm"
    my_sim = Simulation(
        network,
        hosts[network],
        experiment_name,
        iterations=1,
        fallow_transponders=n_ftx,
        te_method="-" + te_method,
        traffic_file=traffic_file,
        fallow_tx_allocation_strategy="static",
        topology_programming_method=tp_method,
        congestion_threshold_upper_bound=0.99999,
        congestion_threshold_lower_bound=0.99999,
        scale_down_factor=SCALE_DOWN_FACTOR, 
        fallow_tx_allocation_strategy="dynamic", 
        top_k=top_k
    )
    demand_factor = float(scale) #* mcf_loss_factor[network][t_class]
    
    file_of_var = { "Congestion": "MaxExpCongestionVsIterations.dat",
                    "Loss":"CongestionLossVsIterations.dat",
                    "Throughput": "TotalThroughputVsIterations.dat",
                    "n_solutions": "TotalSolutions.dat",
                    "doppler_min_mlu": "DopplerMinMLU.dat",
                    "opt_time": "OptTime.dat",
                    "Current Topology ID": "CurrTopoID.dat",
                    "Optimal Topology ID": "OptimalTopoID.dat"
                   }
    tracked_vars = list(file_of_var.keys())
    result = {}

    try: 
        assert use_cached_result, "Ignoring cached result. Starting new experiment"
        result, expected_res_path = check_complete(my_sim, tracked_vars, result)        
    except FileNotFoundError as e:
        expected_res_path = e.filename
        result = my_sim.perform_sim(
            demand_factor=demand_factor, repeat=repeat[tp_method]
        )
    except AssertionError:
        expected_res_path = get_result_path(my_sim)
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
        if curr_res_path == expected_res_path:
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
                    else:
                        if tv == tracked_vars[-1]:
                            report_fob.write(f"NaN\n")
                        else:
                            report_fob.write(f"NaN,")
            logger.info(f"Wrote report to {report_path}")
            return report_path
        else:
            logger.error(f"After sim, result path does not match. Expected: {expected_res_path} got: {curr_res_path}")
    else:
        logger.error("Did not have a result to write. Execution of sim failed.")


def concat_reports(experiment_signatures):
    from glob import glob
    def _concat_reports(reports_glob): 
        summary_file = reports_glob[:-1] + ".csv"
        # summary_data = "Network,Traffic,Demand Scale,TE,TP,Max Link Utilization,Loss,Throughput,Total Solutions,Doppler Min MLU,Opt Time\n"
        summary_data = "Network,Traffic,Demand Scale,TE,TP,Max Link Utilization,Loss,Throughput,Total Solutions,Doppler Min MLU,Optimization Time,Topology ID,Optimal Topology ID\n"
        for f in sorted(glob(reports_glob)):
            with open(f, 'r') as fob: 
                summary_data += "".join(fob.readlines())
        
        with open(summary_file, 'w') as fob: 
            fob.write(summary_data)
        print(f"Summarized report in: {summary_file}")

    for es in experiment_signatures:
        _concat_reports("data/reports/"+ '-'.join(es) + '*')


DEBUG = False
RERUN_OK = True
# PARALLEL = True
PARALLEL = False
def main():
    # network = "Tinet"
    # traffic = "background"
    # scale = "1.4"
    # te = "semimcfraekeft"
    # tp = "greylambda"

    network = ["Campus"]
    # network = ["linear_3"]

    # network = ["Regional"]
    # network = ["four-node"]
    # network = "areon"
    traffic = ["background"]
    # scale = ["0.5"]
    # scale = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    # scale = ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "2.0"]
    # scale = [str(i/10) for i in range(1,31) ]
    # scale = [str(i/10) for i in range(25,28) ]    
    scale = ["1"]
    te = ["ecmp"]
    tp = ["Doppler"]
    top_k = [90, 80, 70, 60, 50, 40, 30, 20, 10]
    n_ftx = [1, 2, 3]
    use_cached_result = [True]
    # time_limit = [60]
    # sol_limit = [1]
    experiment_params = product(network, traffic, scale, te, tp, use_cached_result, top_k, n_ftx)

    if PARALLEL: 
        pool = multiprocessing.Pool()
        pool.map_async(
            experiment, 
            experiment_params)
        pool.close()
        pool.join()
    else:
        for e in experiment_params:
            experiment(e)
            # break                                       

    concat_reports(product(te, tp, network, traffic))
    
    # args = ("Comcast","background","0.8","mcf","greylambda")
    # args = ("Comcast","background-plus-flashcrowd","0.3","mcf","greylambda")
    # experiment(*args)

if __name__ == "__main__":
    main()