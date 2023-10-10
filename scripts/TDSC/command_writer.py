from collections import defaultdict
from glob import glob


# network...................... Network Name
# te_method.................... Traffic Engineering Scheme ('-ecmp', '-mcf', etc.)
# traffic_type................. Description of TM (no spaces) Only used for file descriptors.
# repeat....................... If "repeat" process traffic matrix with and without GreyLambda
# traffic_file................. File name (path) for traffic matrix
# n_fallow_transponders........ Number of fallow transponders per node (ignored but required if 'fallow_tx_allocation'=='file' )
# optical_strategy............. 'onset' 'graylambda' or 'baseline'
# fallow_tx_allocation......... 'static', 'dynamic', or 'file'

def get_post_failures_tm2topo_dicts():
    # topo_meta_file    = "/home/mhall/OLTE/data/graphs/fiber_cut/{}/cut_scenario_meta.tsv".format(network)
    topo_meta_file = "/home/mhall/OLTE/data/graphs/gml/{}_fiber_cut/cut_scenario_meta.tsv".format(
        network)
    traffic_meta_file = "/home/mhall/OLTE/data/traffic/{}_flash_crowd_meta.csv".format(
        network)

    TM_d = defaultdict(str)  # TM ID -> Link
    topo_dict = defaultdict(str)  # Link -> Topo ID

    with open(traffic_meta_file, 'r') as tm_fob:
        # skip first line (header)
        # tm_fob.readline()
        for line in tm_fob:
            ID, link = line.strip().split(';')
            TM_d[ID] = link

    with open(topo_meta_file, 'r') as topo_fob:
        # skip first line (header)
        topo_fob.readline()
        for line in topo_fob:
            ID, link = line.strip().split(';')
            # reverse link to capture traffic that uses reverse path.
            link_reverse = "(" + \
                ",".join(list(reversed(link.strip('()').split(','))))+")"
            topo_dict[link] = ID
            topo_dict[link_reverse] = ID

    return TM_d, topo_dict

def get_network_instance_list():
    if failure == "True":
        TM_d, topo_dict = get_post_failures_tm2topo_dicts()
        topo_id = [topo_dict[TM_d[str(i)]] for i in range(1, iterations+1)]
        return ["{0}_fiber_cut/{0}_{1}_1_edges".format(network, tid) for tid in topo_id]

    else:
        return [network for _ in range(iterations)]

def write_commands():
    commands = []
    for i in range(1, iterations+1):
        commands.append("eval_scripts/sim_event.py " +
                        " ".join([network,
                                  hosts,
                                  te_method,
                                  traffic_type,
                                  str(i),
                                  experiment_tag,
                                  network_instance_list[i-1],
                                  repeat,
                                  traffic_file,
                                  n_fallow_transponders,
                                  optical_strategy,
                                  fallow_tx_allocation,
                                  ftx_file]) + "\n")
    traffic_short_name = traffic_file.split("/")[-1].split('.')[0]
    with open(f"eval_scripts/args/net-{network}" +
              f"_TE{te_method}" +
              f"_-trafficDscription-{traffic_type}" +
              f"_repeat-{repeat}" +
              f"_failure-{failure}" +
              f"_trafficFile-{traffic_short_name}" +
              f"_fallowTXP-{n_fallow_transponders}" +
              f"_opticalStrategy-{optical_strategy}" +
              f"_FTXAllocPolicy-{fallow_tx_allocation}", 'w') as fob:
        fob.writelines(commands)

def get_hosts():
    import networkx as nx
    topology_file = f"data/graphs/gml/{network}.gml"
    G = nx.read_gml(topology_file)
    return str(len(G.nodes()))

def get_iterations():    
    return sum(1 for _ in open(traffic_file))    

if __name__ == "__main__":
    from sys import argv
    if len(argv) == 1:
        network = 'Comcast'
        te_method = "-semimcfraeke"
        traffic_type = 'FlashCrowd'
        repeat = "repeat"
        failure = "True"
        traffic_file = "/home/mhall/OLTE/data/traffic/Verizon_coremelt_every_link_2.00e+11.txt"
        n_fallow_transponders = "20"
        optical_strategy = "greylambda"
        fallow_tx_allocation = "static"

    else:
        try:
            _,                      \
            network,                \
            te_method,              \
            traffic_type,           \
            repeat,                 \
            failure,                \
            traffic_file,           \
            n_fallow_transponders,  \
            optical_strategy,       \
            fallow_tx_allocation    = argv
        except:
            print(
                f"USAGE: {argv[0]} network te_method traffic_type repeat failure traffic_file n_fallow_transponders optical_strategy fallow_tx_allocation")            
            print(
                '''
                
                # network...................... Network Name
                # te_method.................... Traffic Engineering Scheme ('-ecmp', '-mcf', etc.)
                # traffic_type................. Description of TM (no spaces) Only used for file descriptors.
                # repeat....................... If "repeat" process traffic matrix with and without GreyLambda
                # traffic_file................. File name (path) for traffic matrix
                # failure...................... Link failure experiment? (True or False)
                # n_fallow_transponders........ Number of fallow transponders per node (ignored but required if
                #                                   'fallow_tx_allocation'=='file' )
                # optical_strategy............. 'onset' 'graylambda' or 'baseline'
                # fallow_tx_allocation......... 'static', 'dynamic', or 'file'

                ''')
            print(
                f"EXAMPLE: {argv[0]} Comcast -semimcfraeke FlashCrowd repeat True data/traffic/Verizon_coremelt_every_link_2.00e+11.txt 20 greylambda static")
            exit(-1)

    experiment_tag = '{}_{}'.format(optical_strategy, fallow_tx_allocation)
    network_instance_list = []

    hosts = get_hosts()
    iterations = get_iterations()

    if fallow_tx_allocation.lower() == "file":
        # ftx file is assumed to be in data/txp.
        for ftx_file in glob(f'data/txp/{network}*'):
            n_fallow_transponders = ftx_file.split("_")[1].split(".")[0]
            network_instance_list = get_network_instance_list()
            write_commands()
    else:
        network_instance_list = get_network_instance_list()
        ftx_file="NULL"
        write_commands()
