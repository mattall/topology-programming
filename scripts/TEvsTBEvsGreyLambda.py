import sys

# sys.path.insert(0, "/home/m/src/topology-programming/src/")
# from onset.simulator import Simulation
from itertools import product
from os import path
import multiprocessing
from experiment_mapped import experiment, experiment_mapped
from experiment_params import *
from datetime import datetime
from sys import argv
pool = multiprocessing.Pool()


if __name__ == "__main__":

    tp_methods = [argv[1]]
    print(tp_methods)
    def custom_error_callback(error):
        print(f"Got an Error: {error}\n", flush=True)

    def custom_callback(result_file):
        print(f"Wrote result to: {result_file}\n", flush=True)

    experiment_combinations = list(product(
        te_methods, tp_methods, networks, t_classes, demand_scale
    ))
    experiment_combinations = [

        ("mcf","greylambda","Comcast","background","0.8"),
        ("mcf","greylambda","Comcast","background-plus-flashcrowd","0.3")]

    print(len(experiment_combinations))
    result_files = sorted(
                        list(
                            pool.map_async(
                                experiment_mapped, 
                                experiment_combinations, 
                                callback=custom_callback, 
                                error_callback=custom_error_callback).get()
                        )
                    )
    pool.close()
    pool.join()

    ts = datetime.isoformat(datetime.now()).replace('.', '_').replace(':', '_')

    with open( f"data/reports/{ts}_demand_scale_results.csv", 'w') as write_fob:        
        for i, rf in enumerate(result_files):
            with open(rf, 'r') as read_fob:
                lines = read_fob.readlines()
                # if i == 0:
                for l in lines:
                    write_fob.write(l)
                # else:
                #     for l in lines[1:]:
                #         write_fob.write(l)


# for (
#     te_method,
#     tp_method,
#     network,
#     t_class,
#     scale,
# ) in experiment_combinations:
#     pool.apply_async(
#         experiment,
#         args=(
#             te_method,
#             tp_method,
#             network,
#             t_class,
#             scale,
#         ),
#         error_callback=custom_error_callback,
#         callback=custom_callback,
#     )