import copy
from collections import defaultdict
from sys import path as sys_path
from os import chdir, curdir
from os import path as os_path
import onset
from onset.utilities.post_process import read_result_val
from glob import iglob

from itertools import product
import matplotlib as mpl
# mpl.use('TKAgg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

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

plt.close('all')

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
    
ECMP_Pallet = [sns.color_palette("tab10")[1], sns.color_palette("tab10")[0]]
MCF_Pallet = sns.color_palette("tab10")[2:4]

Networks=["ANS", "CRL", "sprint", "bellCanada", "surfNet"]
Defense=["Baseline", "ONSET"]
Routings=["ecmp", "mcf"]

network_names = {"bellCanada": "Bell Canada", "sprint": "Sprint", "surfNet": "Surf Net", "ANS": "ANS", "CRL": "CRL"}
baseline_label = {"ecmp": "ECMP", "mcf": "Ripple*"} 
onset_label =  {"ecmp": "ECMP+ONSET" , "mcf": "Ripple*+ONSET"}


for _network, routing in product(Networks, Routings):
    network = network_names[_network]
    plt.clf()
    baseline_series = {}
    # for i, f in enumerate(iglob(f"data/archive/rolling-02-01-24/{network}*{routing}*baseline*/*_10/MaxExpCongestion*")):
    for i, f in enumerate(iglob(f"data/archive/rolling-02-01-24/results/{_network}_{_network}*{routing}*baseline*/*s_10/MaxExpCongestionVsIterations.dat")):        
        if i == 720:
            break
        iter_i = int(f.split('/')[-2].split('_')[1].split('-')[0]) #"data/archive/rolling-02-01-24/ANS_ANS_-ecmp_baseline_rolling_mixed_type_attackcsv_static_2_baseline_max_1__-ecmp_100/ANS_132-0-720_1_0_None_0_Gbps_10/MaxExpCongestionVsIterations.dat"
        baseline_series[iter_i] = read_result_val(f)
    onset_series = {}
    
    # for i, f in enumerate(iglob(f"data/archive/rolling-02-01-24/{network}*{routing}*baseline*/*_10/MaxExpCongestion*")):
    for i, f in enumerate(iglob(f"data/archive/rolling-02-01-24/results/{_network}_{_network}*{routing}*onset*/*s_10/MaxExpCongestionVsIterations.dat")):        
        if i == 720:
            break
        iter_i = int(f.split('/')[-2].split('_')[1].split('-')[0]) #"data/archive/rolling-02-01-24/ANS_ANS_-ecmp_baseline_rolling_mixed_type_attackcsv_static_2_baseline_max_1__-ecmp_100/ANS_132-0-720_1_0_None_0_Gbps_10/MaxExpCongestionVsIterations.dat"
        onset_series[iter_i] = read_result_val(f)

    plt.rc('font', size=22)    
    baseline_label = {"ecmp": "ECMP", "mcf": "Ripple*"} 
    onset_label =  {"ecmp": "ECMP+ONSET" , "mcf": "Ripple*+ONSET"}        
    
    baseline_series_list = sorted(list(baseline_series.items()))
    baseline_iterations = [x for x, _ in baseline_series_list]
    baseline_congestion = [y for _, y in baseline_series_list]
    baseline_defense = [baseline_label[routing]] * len(baseline_series)

    onset_series_list = sorted(list(onset_series.items()))
    onset_iterations = [x for x, _ in onset_series_list]
    onset_congestion = [y for _, y in onset_series_list]
    onset_defense = [onset_label[routing]] * len(onset_series)

    fig, ax = plt.subplots(figsize=(8,4))
    if routing == 'ecmp': 
        pallet = sns.color_palette("tab10")[0:2]
    if routing == 'mcf':
        pallet = sns.color_palette("tab10")[2:4]
    # lines = sns.lineplot(data=df, x=df["Iteration"], y=df['Congestion'], hue=df["Strategy"], style_order=style_order)
    sns.lineplot(x=baseline_iterations, y=baseline_congestion, color=pallet[0], linestyle='-', linewidth=3, ax=ax)
    sns.lineplot(x=onset_iterations, y=onset_congestion, color=pallet[1], linestyle='--', linewidth=3, ax=ax)
    
    my_legend = plt.legend(title=None, bbox_to_anchor=(-0.2, 2.0), loc='upper left', borderaxespad=0, ncol=2, frameon=False)

    ax.set_xticks([x*120 for x in range(7)])
    
    ax.set_xticklabels([0,10,20,30,40,50,60])
    ax.set_xlabel("Time (m)")

    ax.set_yticks([0, 1, 2])
    # lines.set_yticklabels([0,10,20,30,40,50,60])

    plt.axhline(1, color="black", linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    # plt.legend()


    # copy the handles
    handles = [copy.copy(ha) for ha in handles ]
    # set the linewidths to the copies
    [ha.set_linewidth(3) for ha in handles ]
    # put the copies into the legend
    # leg = plt.legend(handles=handles, labels=labels)
    

    # my_legend = plt.legend([h for h in reversed(handles)], [baseline_label[routing], onset_label[routing]], bbox_to_anchor=(-0.2, 2.0), loc='upper left', borderaxespad=0, ncol=2, frameon=False)
    # my_legend = plt.legend(handles, ["ECMP+ONSET", "ECMP"], bbox_to_anchor=(-0.2, 2.0), loc='upper left', borderaxespad=0, ncol=2)
    if routing == 'mcf':
        figure_name = f"data/plots/rolling/rolling_attack_congestion_MCF_{_network}".replace(" ", "_").replace('*', '_')    
    else:
        figure_name = f"data/plots/rolling/rolling_attack_congestion_{baseline_label[routing]}_{_network}".replace(" ", "_").replace('*', '_')    
    # export_legend(my_legend, f"{figure_name}_legend.pdf")
    plt.legend().remove()
    # plt.axvline(12*10, color="black", linestyle="--")
    # plt.axvline(12*5, color="black", linestyle="--")
    plt.tight_layout(pad=0)
    plt.savefig(f"{figure_name}.pdf")   
    print(f"{figure_name}.pdf") 
    plt.close('all')
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
