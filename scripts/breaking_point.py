from onset.simulator import Simulation
from os import path
from sys import exit

# network = "sndlib_abilene"
# hosts = 12

# network = "sndlib_geant"
# hosts = 22

# topo = "hdumb"
# topo = "tiered"

# exp_type = "ddos"   
# exp_type = "te"   

# te_method = "-mcf"

# otp = "0ft"
# otp = "2ft"
# otp = "4ft"

# if topo == "hdumb":
#     network = "hdumb"
#     hosts = 17
#     te_method = "-mcf"
#     demand_factor = 5.846151300000004e-05
    
# elif topo == "tiered":
#     network = "tiered"
#     hosts = 26
#     demand_factor = 0.0000001

#     traffic, name
#     params = ("data/traffic/{network}_mixed_attack", "otp_plus_pdp_{exp_type}_{otp}")
    



# traffic = "data/traffic/vtc_hdumb-tm"

# network = "sndlib_germany50"
# hosts = 50

# network = "sndlib_nobel-germany"
# hosts = 17
# traffic = "data/traffic/aggregate_tiered-tm"

# traffic = "data/traffic/tiered_mixed_attack"
# experiment_name = "otp_ddos"

# traffic = "data/traffic/video_tiered-tm"
# experiment_name = "otp_plus_pdp"

# traffic = "data/traffic/aggregate_tiered-tm"
# experiment_name = "otp"


# te_method = "-mcf"

network = "ground_truth"
experiment_name = 

my_sim = Simulation(network,
            hosts, 
            experiment_name,  
            iterations=1, 
            fallow_transponders=2,
            te_method=te_method,
            traffic_file=traffic,
            fallow_tx_allocation_strategy="read_capacity",
            topology_programming_method="greylambda",
            congestion_threshold_upper_bound=0.99999,
            congestion_threshold_lower_bound=0.99999
            )    # 


try:
    report_path = f"data/reports/{network}_{experiment_name}_{te_method}.csv"
    if path.exists(report_path):
        print_header = False
        report_fob = open(report_path, "a")
    else:
        print_header = True
        report_fob = open(report_path, "w")

    prev_path=""
    while True:

        result = my_sim.perform_sim(unit="Mbps", demand_factor=demand_factor, repeat=True)


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
    exit()
        