from networkx import Graph
import copy
from collections import defaultdict
from sys import argv
from sys import path as sys_path
from os import chdir, curdir
from os import path as os_path
import onset
from onset.utilities.post_process import read_result_val
from onset.utilities.graph import read_gml
from glob import glob
from itertools import product
import matplotlib as mpl
# mpl.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import read_dot

def read_dot_to_undirected_core(path): 
    G = read_dot(path).to_undirected()
    G.remove_edges_from([e for e in G.edges() if 'h' in e[0] or 'h' in e[1]])
    return G

def fill_missing(topo_list, net, topo_root):
    for i, topo in enumerate(topo_list): 
        if topo is None: 
            min_max_congestion = 500
            sol_topo = None
            for max_congestion_file in glob(topo_root + net + f"_{i}_*sol*/MaxExpCongestion*"):
                print(max_congestion_file)
                max_congestion_result = read_result_val(max_congestion_file)
                if max_congestion_result < min_max_congestion: 
                    min_max_congestion = max_congestion_result
                    sol_topo = "/".join(max_congestion_file.split('/')[:-1])                
            topo_list[i] = sol_topo


def get_add_drop_series(G_series):
    links_added = []
    links_dropped = []
    links = []
    G_0 = Graph()
    for i, topo in enumerate(G_series):
        G_t = read_dot_to_undirected_core(topo)        
        if i == 0: 
            G_0 = G_t
            links.append(len(G_t.edges))
            links_added.append(0)
            links_dropped.append(0)
        else:
            added = sum( [ 1 for edge in G_t.edges if edge not in G_0.edges] )
            dropped = sum( [ 1 for edge in G_0.edges if edge not in G_t.edges] )
            links.append(links[i-1] + added - dropped)
            links_added.append(added)
            links_dropped.append(dropped)
            G_0 = G_t
    return links_added, links_dropped, links   
# Networks=("ANS CRL sprint bellCanada surfNet")
# Routings=("mcf" "ecmp")
# for network in ${Networks[*]}; do
# for routing in ${Routings[*]}; do
# for scale in ${Scales[*]}; do
# for iter in ${Iters[*]}; do
# echo "cat data/archive/crossfire-rerun-02-01-24/results/${network}*${routing}*${scale}*/*-${iter}-*_10/MaxExpCongestion*"
# # cat data/archive/crossfire-rerun-02-01-24/results/${network}*${routing}*${scale}*/*-${iter}-*_10/MaxExpCongestion* | grep ${routing} | awk '{print $3}' >> data/archive/crossfire-rerun-02-01-24/congestion/${network}-${routing}-${scale}-${iter}.dat; 
# done
# done
# done
# done

HOME = os_path.expanduser("~")
# chdir(f"{HOME}/src/topology-programming")
chdir(f"/home/mhall7/durairajanlab.matt/topology-programming/")

sys_path.insert(0, "src/")
sys_path.insert(0, "src/utilities")

# from onset.utilities import plotters as my_plotters

def export_legend(legend, filename="legend"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())    
    fig.savefig(filename+".pdf", dpi="figure", bbox_inches=bbox)
    print("Saved legend: ", filename+".pdf")

if __name__ == "__main__": 
    ECMP_Pallet = [sns.color_palette("tab10")[1], sns.color_palette("tab10")[0]]
    MCF_Pallet = sns.color_palette("tab10")[2:4]

    Networks=["ANS", "CRL", "sprint", "bellCanada", "surfNet"]
    Defenses=["baseline", "onset"]
    Routings=["ecmp", "mcf"]
    
    network_names = {"bellCanada": "Bell Canada", "sprint": "Sprint", "surfNet": "Surf Net", "ANS": "ANS", "CRL": "CRL"}
    baseline_label = {"ecmp": "ECMP", "mcf": "Ripple*"} 
    onset_label =  {"ecmp": "ECMP+ONSET" , "mcf": "Ripple*+ONSET"}

    defense_label = {"ecmp": "ECMP+ONSET",
                    "mcf": "Ripple*+ONSET"}
    
    # Routings=["mcf"]
    # Networks = ["ANS"]

    for _network in Networks:
        try: 
            df = pd.read_csv(f"data/archive/rolling-02-01-24/links_change_{_network}.csv")
        except:
            df = None
            network = network_names[_network]
            plt.clf()
            topology = {}  
            iterations = [_ for _ in range(720)]  
            for routing in Routings: 
                topology[routing] = [None] * 720
                topo_glob = sorted(glob(f"data/archive/rolling-02-01-24/results/{_network}_{_network}*{routing}*onset*/*s_10.dot"), key=lambda f_: int(f_.split('/')[-1].split('_')[1].
                split('-')[0]))
                topo_root = f"data/archive/rolling-02-01-24/results/{_network}_{_network}*{routing}*onset*/"
                for i, f in enumerate(topo_glob):                
                    iter_i = int(f.split('/')[-1].split('_')[1].split('-')[0]) 
                    if iter_i >= 720:
                        break
                    if topology[routing][iter_i] is None:
                        topology[routing][iter_i] = f
                
                if None in topology[routing]:
                    fill_missing(topology[routing], _network, topo_root)

                defense_measures = get_add_drop_series(topology[routing])
                defense_list = [defense_label[routing]] * len(iterations)
                if df is None:
                    df = pd.DataFrame( {
                                "Iteration" : iterations,
                                "Network": [network_names[_network]] * len(iterations), 
                                "Defense": defense_list, 
                                "Links Added": defense_measures[0], 
                                "Links Dropped": defense_measures[1],
                                "Total Links": defense_measures[2]} )
                else:
                    df = pd.concat([df,pd.DataFrame( {
                                "Iteration" : iterations,
                                "Network": [network_names[_network]] * len(iterations), 
                                "Defense":  defense_list, 
                                "Links Added": defense_measures[0], 
                                "Links Dropped": defense_measures[1],
                                "Total Links": defense_measures[2]} )])
                    df.reset_index()

            df.to_csv(f"data/archive/rolling-02-01-24/links_change_{_network}.csv")
            print("wrote", f"data/archive/rolling-02-01-24/links_change_{_network}.csv")
        # ripple_topology = [None] * 720
        # for i, f in enumerate(sorted(glob(f"data/archive/rolling-02-01-24/results/{_network}_{_network}*{routing}*onset*/*s_10.gml"), key = lambda f_: int(f_.split('/')[-1].split('_')[1].split('-')[0]))):        
        #     iter_i = int(f.split('/')[-1].split('_')[1].split('-')[0]) 
        #     if iter_i >= 720:
        #         break
        #     if ripple_topology[iter_i] is None:    
        #         ripple_topology[iter_i] = f
        # fill_missing(ecmp_iterations)
                

        measures = ("Links Added", "Links Dropped", "Total Links")    
        

        # ripple_measures = get_add_drop_series(ripple_topology)
        # ripple_defense = [onset_label[routing]] * len(ripple_iterations)

        # df = pd.concat([df, pd.DataFrame( {"Network": [_network] * len(iterations), 
        #                         "Defense": ripple_defense, 
        #                         "Links Added": ripple_measures[0], 
        #                         "Links Dropped": ripple_measures[1],
        #                         "Total Links": ripple_measures[2]})
        #                         ], ignore_index=True)

        
        # baseline_label = {"ecmp": "ECMP", "mcf": "Ripple*"} 
        # onset_label =  {"ecmp": "ECMP+ONSET" , "mcf": "Ripple*+ONSET"}        
        for measure in measures:
            plt.rc('font', size=22)
            fig, ax = plt.subplots(figsize=(8,4))
            
            palette =  [sns.color_palette("tab10")[0],sns.color_palette("tab10")[2]]
            
            # lines = sns.lineplot(data=df, x=df["Iteration"], y=df['Congestion'], hue=df["Strategy"], style_order=style_order)
            sns.lineplot(data=df, x="Iteration", y=measure, hue="Defense", palette=palette, style="Defense", linewidth=3, ax=ax)
            # sns.lineplot(x=iterations, y=ecmp, color=ECMP_Pallet[0], linestyle='-', linewidth=3, ax=ax)
            # sns.lineplot(x=iterations, y=ripple, color=MCF_Pallet[1], linestyle='--', linewidth=3, ax=ax)
            
            
            ax.set_xticks([x*120 for x in range(7)])
            ax.set_xticklabels([0,10,20,30,40,50,60])
            ax.set_xlabel(measure)
            ax.set_ylim(0)
            ax.set_xticks([x*120 for x in range(7)])

            # ax.set_yticks([0, 1, 2])
            # plt.axhline(1, color="black", linestyle="--")
            # handles, labels = ax.get_legend_handles_labels()    
            # handles = [copy.copy(ha) for ha in handles ]    
            # [ha.set_linewidth(3) for ha in handles ]        

            figure_name = f"data/plots/rolling/rolling_attack_{_network}_{measure}".replace(" ", "_").replace('*', '_')        
            # my_legend = plt.legend(title=None, bbox_to_anchor=(-0.2, 2.0), loc='upper left', borderaxespad=0, ncol=2, frameon=False)
            # export_legend(my_legend, figure_name + "_legend")
            # plt.legend().remove()    
            # plt.close()
            # exit()
            plt.tight_layout(pad=0)
            plt.savefig(f"{figure_name}.pdf")   
            print(f"{figure_name}.pdf") 
            plt.close()
    # # ax2 = sns.ecdfplot(data=data, x="Congestion", hue="Strategy", linewidth=3, palette=palette)
    # # Set up the initial legend.                    
    # # my_legend = plt.legend(legend_key, bbox_to_anchor=(-0.2, 2), loc='upper left', borderaxespad=0, ncol=2, frameon=False)
    # # my_legend = plt.legend(legend_key,  bbox_to_anchor=(-0.2, 2), loc='upper left', borderaxespad=0, ncol=2, frameon=False)  
    # # handles, labels = my_legend.get_legend_handles_labels()
    # plt.tick_params(length=8)
    # # export_legend(my_legend, figurename + "_legend")
    # plt.legend().remove()
    # plt.axvline(1, linestyle='--', color='black', linewidth=2)
    # plt.ylabel("CDF")
    # plt.xlabel("Max. Congestion")
    # plt.tight_layout()
    # plt.savefig(figure_name + ".pdf")
    # print(f"saved: {figure_name}.pdf")
    # plt.clf()
    # plt.close('all')
    # break
    # with open(figure_name + "_stats.txt", 'w') as fob:
    #     fob.write("\t\t\t,Network,\tLoss Events,\ttotal\n")
    #     fob.write(f"{routing},\t{network},\t{len(baseline_df[baseline_df['Loss'] > 0])},\t{len(baseline_df)}\n")
    #     fob.write(f"{routing}+ONSET,\t{network},\t{len(onset_df[onset_df['Loss'] > 0])},\t{len(onset_df)}")
    # print(f"saved: {figure_name}_stats.txt")
        # print( data[["Network", "Strategy", "Routing", "Congestion"]].sort_values(by="Strategy") )
    
#         iters_complete.append(iter_i)
#         # print(i, iter_i)

#     for i in range(0, 720): 
#         if i not in iters_complete: 
#             iters_missing.append(i)
#     iters_missing = sorted(iters_missing)
#     print(f"\nNetwork: {_network},\tRouting: {routing}\tDefense: Baseline ")
#     if (len(iters_complete) == 720):
#         print(f"Work complete.\n")
#     elif (len(iters_complete) > 0):
#         print(f"Work complete.\n")
#         print(f"last iter complete: {max(iters_complete)}")
#         print(f"First 5 missing iters: {iters_missing[:5]}")
#     else:
#         print(f"Not started")
#         print(f"First 5 missing iters: {iters_missing[:5]}")
#     # need to order the time series data. 
#     if iters_missing:
#         fob.write(f"('{_network}', '-{routing}', 'baseline') : {iters_missing[0]}, \n")
#     else: 
#         fob.write(f"('{_network}', '-{routing}', 'baseline') : 720, \n")
#     iters_complete = []
#     iters_missing = []
#     for i, f in enumerate(iglob(f"data/results/{network}_{network}*{routing}*onset*/*s_10/MaxCongestionVsIterations.dat")):        
#     # for i, f in enumerate(iglob(f"data/archive/rolling-02-01-24/{network}_{network}*{routing}*onset*/*s_10/MaxCongestionVsIterations.dat")):        
#         iter_i = int(f.split('/')[-2].split('_')[1].split('-')[0]) #"data/archive/rolling-02-01-24/ANS_ANS_-ecmp_baseline_rolling_mixed_type_attackcsv_static_2_baseline_max_1__-ecmp_100/ANS_132-0-720_1_0_None_0_Gbps_10/MaxExpCongestionVsIterations.dat"
#         iters_complete.append(iter_i)
#         # print(i, iter_i)

#     for i in range(0, 720): 
#         if i not in iters_complete: 
#             iters_missing.append(i)
#     iters_missing = sorted(iters_missing)
#     print(f"\nNetwork: {_network},\tRouting: {routing}\tDefense: onset_v3 ")
#     if (len(iters_complete) == 720):
#         print(f"Work complete.\n")
#     elif (len(iters_complete) > 0):
#         print(f"Work complete.\n")
#         print(f"last iter complete: {max(iters_complete)}")
#         print(f"First 5 missing iters: {iters_missing[:5]}")
#     else:
#         print(f"Not started")
#         print(f"First 5 missing iters: {iters_missing[:5]}")
#     if iters_missing:
#         fob.write(f"('{_network}', '-{routing}', 'onset_v3') : {iters_missing[0]},\n")    
#     else:
#         fob.write(f"('{_network}', '-{routing}', 'onset_v3') : 720,\n")    
# fob.write("}")
# fob.close()
# exit()
