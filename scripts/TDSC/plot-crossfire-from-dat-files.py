from collections import defaultdict
from sys import path as sys_path
from os import chdir, curdir
from os import path as os_path

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
# Scales=("100E9" "200E9")
# Strategies=("baseline" "onset")
# for network in ${Networks[*]}; do
# for routing in ${Routings[*]}; do
# for scale in ${Scales[*]}; do
# for iter in ${Iters[*]}; do
# echo "network"
# # #  ls -l data/results/${network}_${network}*${routing}*${strat}*/*s_10/MaxCongestionVsIterations.dat | wc -l
# # cat data/archive/crossfire-rerun-02-01-24/results/${network}*${routing}*${scale}*/*-${iter}-*Gbps_10/MaxExpCongestion* | grep ${routing} | awk '{print $3}' > data/archive/crossfire-rerun-02-01-24/congestion/${network}-${routing}-${scale}-${iter}.dat; 
# ls data/archive/crossfire-rerun-02-01-24/results/${network}*${routing}*${scale}*/*Gbps_10/MaxExpCongestion* | wc -l
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
Routings=["mcf", "ecmp"]
Scales=["100E9", "200E9"]
Iters=["0", "1"]

network_names = {"bellCanada": "Bell Canada", "sprint": "Sprint", "surfNet": "Surf Net", "ANS": "ANS", "CRL": "CRL"}

with open("data/plots/crossfire/congestion_stats.csv", 'w') as fob:
    fob.write("Routing,Network,Baseline Loss Events,Baseline Events,Baseline Ratio,ONSET Loss Events,ONSET events,ONSET Ratio\n")

    for _network, scale, routing in product(Networks, Scales, Routings):
        network = network_names[_network]
        plt.clf()
        baseline_f = f"data/archive/crossfire-rerun-02-01-24/congestion/{_network}-{routing}-{scale}-0.dat"
        onset_f = f"data/archive/crossfire-rerun-02-01-24/congestion/{_network}-{routing}-{scale}-1.dat"

        baseline = np.loadtxt(baseline_f)
        onset = np.loadtxt(onset_f)
        if 0: 
            print(routing)
            if routing == "ecmp": 
                palette = ECMP_Pallet
                legend_key = ["ECMP+ONSET" , "ECMP"]
            if routing == "mcf":
                palette = MCF_Pallet                
                legend_key = ["Ripple*+ONSET", "Ripple*" ]
            plt.rc('font', size=22)    
            baseline_label = {"ecmp": "ECMP", "mcf": "Ripple*"} 
            onset_label =  {"ecmp": "ECMP+ONSET" , "mcf": "Ripple*+ONSET"}
            figure_name = f"data/plots/crossfire/congestion-{network}-{baseline_label[routing]}-{scale}".replace(" ", "_").replace('*', '_')
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()
            ax = sns.ecdfplot(data=onset, linewidth=3, label=onset_label[routing], color=palette[0])
            ax1 = sns.ecdfplot(data=baseline, linestyle="--", linewidth=3, label=baseline_label[routing], color=palette[1])
            plt.legend(title=None)
            # # ax2 = sns.ecdfplot(data=data, x="Congestion", hue="Strategy", linewidth=3, palette=palette)
            # Set up the initial legend.                                
            my_legend = plt.legend(bbox_to_anchor=(-0.2, 2), loc='upper left', borderaxespad=0, ncol=2, frameon=False)  
            # handles, labels = my_legend.get_legend_handles_labels()
            plt.tick_params(length=8)
            export_legend(my_legend, figure_name + "_legend")
            plt.legend().remove()
            plt.axvline(1, linestyle='--', color='black', linewidth=2)
            plt.ylabel("CDF")
            plt.xlabel("Max. Congestion")
            plt.tight_layout(pad=0)
            plt.savefig(figure_name + ".pdf")
            print(f"saved: {figure_name}.pdf")
            plt.clf()
            plt.close('all')
        
        fob.write(f"{routing},{network},{len(baseline[baseline > 1])},{len(baseline)},{len(baseline[baseline > 1])/len(baseline)},{len(onset[onset > 1])},{len(onset)},{len(onset[onset > 1])/len(onset)}\n")
        
        
        # print( data[["Network", "Strategy", "Routing", "Congestion"]].sort_values(by="Strategy") )
        