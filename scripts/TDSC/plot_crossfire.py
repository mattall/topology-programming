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

plt.close('all')

HOME = os_path.expanduser("~")
chdir(f"{HOME}/src/topology-programming")

sys_path.insert(0, "src/")
sys_path.insert(0, "src/utilities")

# from onset.utilities import plotters as my_plotters

report_folder = "data/archive/crossfire-rerun-02-01-24/reports/"

def export_legend(legend, filename="legend"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename+".pdf", dpi="figure", bbox_inches=bbox)
    
# export_legend(my_legend, figurename + "_legend")
# plt.legend().remove()
ECMP_Pallet = [sns.color_palette("tab10")[1], sns.color_palette("tab10")[0]]
MCF_Pallet = sns.color_palette("tab10")[2:4]

# Parameter dictionary
# plt.rcParams

result_df = pd.DataFrame()
all_columns = set()
for data_file in iglob(report_folder + "*.csv"):
    df = pd.read_csv(data_file)    
    all_columns.update(df.columns)
    result_df = pd.concat([result_df, df], ignore_index=True)

result_df = result_df.sort_values(by=["Experiment", "Iteration"])
result_df = result_df.reindex(columns=all_columns)
# result_df.sort_values("Experiment")
# result_df.reindex()
result_df.to_csv("data/reports/crossfire.csv")



networks = {"bellCanada": "Bell Canada", "sprint": "Sprint", "surfNet": "Surf Net", "ANS": "ANS", "CRL": "CRL"}
result_df["Network"] = result_df["Experiment"].apply(lambda x: x.split("_")[0] if isinstance(x, str) else x)
result_df["Network"] = result_df["Network"].apply(lambda x: networks[x])

result_df["Routing"] = result_df["Routing"].apply(lambda x: "Ripple*" if x == "MCF" else x)

result_df["Strategy"] = result_df.apply(
    lambda row: row["Routing"] if row["Iteration"].split("-")[-1] == "0" else f"{row['Routing']}+ONSET", axis=1
)
result_df["Attack Scale"] = result_df["Experiment"].apply(lambda x: "100E9" if "100E9" in x else "200E9")

# scale = "100E9"
# network = "Bell Canada"
# routing = "ECMP"
# print(list(product(result_df["Network"].unique(), result_df["Attack Scale"].unique(), result_df["Routing"].unique())))

for network, scale, routing in product(result_df["Network"].unique(), result_df["Attack Scale"].unique(), result_df["Routing"].unique()):
    
    if routing == "ECMP": continue
    print(network, scale, routing)
    temp = result_df[result_df["Network"] == network]
    
    temp = temp[temp["Attack Scale"] == scale]
    # print(temp[["Network", "Strategy", "Routing", "Congestion"]])
    data = temp[temp["Routing"] == routing]


    # print(temp[["Network", "Strategy", "Routing", "Congestion"]])
    baseline_df = data[data["Strategy"] == routing]
    onset_df = data[data["Strategy"] == routing + "+ONSET"]

    # print(data[["Network", "Strategy", "Routing", "Congestion"]])    

    if routing == "ECMP": 
        pallet = ECMP_Pallet
        legend_key = ["ECMP", "ECMP+ONSET" ]
    if routing == "Ripple*":
        pallet = MCF_Pallet        
        legend_key = ["Ripple*", "Ripple*+ONSET" ]

    label = {
        "ECMP":
        "MCF"
    }
    # baseline_df = np.array(data_1["Congestion"])
    # onset_df = np.array(data_2["Congestion"])

    # data_dict = {"ECMP": X_1, "ECMP+ONSET": X_2}

    # my_plotters.congestion_multi_cdf(
    #     data_dict, inLabel="Max. Link Utilization", fig_name=f"data/plots/crossfire/congestion-{network}-{routing}-{scale}"
    # )


    plt.rc('font', size=22)
    figurename = f"data/plots/crossfire/congestion-{network}-{routing}-{scale}".replace(" ", "_")
    # )

    # ax1 = sns.ecdfplot(data=baseline_df, x="Congestion", linestyle="--", linewidth=3, color = pallet[0], labels="")
    # ax2 = sns.ecdfplot(data=onset_df, x="Congestion", linewidth=3, color = pallet[1] )

    ax2 = sns.ecdfplot(data=data, x="Congestion", hue="Strategy", linewidth=3)
    # Set up the initial legend.                    
    # my_legend = plt.legend(legend_key, bbox_to_anchor=(-0.2, 2), loc='upper left', borderaxespad=0, ncol=2, frameon=False)
    # handles, labels = my_legend.get_legend_handles_labels()
    plt.tick_params(length=8)
    # export_legend(my_legend, figurename + "_legend")
    # plt.legend().remove()
    plt.axvline(1, linestyle='--', color='black', linewidth=2)
    plt.ylabel("CDF")
    plt.xlabel("Max. Congestion")
    plt.tight_layout()
    plt.savefig(figurename + ".pdf")
    print(f"saved: {figurename}.pdf")
    plt.clf()
    plt.close()
    with open(figurename + "_stats.txt", 'w') as fob:
        fob.write("\t,Network,\tLoss Events,\ttotal\n")
        fob.write(f"{routing},\t{network},\t{len(baseline_df[baseline_df['Loss'] > 0])},\t{len(baseline_df)}\n")
        fob.write(f"{routing}+ONSET,\t{network},\t{len(onset_df[onset_df['Loss'] > 0])},\t{len(onset_df)}")
    print(f"saved: {figurename}_stats.txt")
    if routing == "Ripple*":
        print( data[["Network", "Strategy", "Routing", "Congestion"]].sort_values(by="Strategy") )
        break