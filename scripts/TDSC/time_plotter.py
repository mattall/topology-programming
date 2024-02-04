import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numbers import Number
import numpy as np
from sys import path as sys_path
from os import path as os_path
from os import chdir
Networks=("ANS CRL sprint bellCanada surfNet")
Routings=("mcf" "ecmp")
Scales=("100E9" "200E9")
Iters=("0" "1")

# for network in ${Networks[*]}; do
# for routing in ${Routings[*]}; do
# for scale in ${Scales[*]}; do
# for iter in ${Iters[*]}; do
# # echo "cat data/archive/crossfire-rerun-02-01-24/results/${network}*${routing}*${scale}*/*-${iter}-*_10/MaxExpCongestion*"
# cat data/archive/crossfire-rerun-02-01-24/results/${network}*${routing}*${scale}*/*-${iter}-*_10/OptTime* | grep ${routing} | awk '{print $3}' >> data/archive/crossfire-rerun-02-01-24/optTime/${network}-${routing}-${scale}-${iter}.dat; 
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

from onset.utilities import plotters as my_plotters
def ecdf(data, array: bool=True):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    if not array:
        return pd.DataFrame({'x': x, 'y': y})
    else:
        return x, y

def export_legend(legend, filename="legend"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename+".pdf", dpi="figure", bbox_inches=bbox)
    
Networks=["ANS", "CRL", "sprint", "bellCanada", "surfNet"]
Routings=["mcf", "ecmp"]
Scales=["100E9", "200E9"]
Iters=["0", "1"]

max_len = 0



network_names = {"bellCanada": "Bell Canada", "sprint": "Sprint", "surfNet": "Surf Net", "ANS": "ANS", "CRL": "CRL"}
opt_time = {}
for _network in Networks: 
    network = network_names[_network]    
    opt_time_f = f"data/archive/crossfire-rerun-02-01-24/optTime/{_network}.dat"
    opt_time[network] = []
    this_len = 0
    with open(opt_time_f, 'r') as fob: 
        for item in fob.readlines():
            
            this_item = item.strip()
            if this_item.replace('.','').isdigit():
                this_len += 1                
                opt_time[network].append(float(this_item))
            if this_len > max_len:
                max_len = this_len

# for net in opt_time: 
#     diff = max_len - len(opt_time[net])
#     if diff > 0: 
#         opt_time[net] += ["NaN"] * diff
#     print(net)
#     print(len(opt_time[net]))
dash_styles = ['-', ':', '--', '-.', (0, (3, 1, 1, 1, 1, 1))]

figure_name = f"data/plots/crossfire/optTime".replace(" ", "_").replace('*', '_')
# df = DataFrame(opt_time)
# df = df.dropna()
plt.subplots(figsize=(6,4))
# Plot CDF for each column
for i, net in enumerate(opt_time):
    sns.ecdfplot(data=opt_time[net], label=net, linestyle=dash_styles[i % len(dash_styles)])
    X, Y = ecdf(opt_time[net], True)
    with open(figure_name + f"_{net}_cdf.csv", 'w') as fob: 
        fob.write(f"Network,Optimization Time\n")
        for x, y in zip(X, Y):
            fob.write(f"{x},{y}\n")

# Set labels and legend
plt.xlabel('Optimization Time')
plt.ylabel('CDF')
my_legend = plt.legend(bbox_to_anchor=(-0.5, .5), loc='center right')
plt.tick_params(length=8)
export_legend(my_legend, figure_name + "_legend")
plt.legend().remove()
plt.tight_layout(pad=0)
plt.savefig(figure_name + ".pdf")
print(f"saved: {figure_name}.pdf")
plt.clf()
plt.close('all')

# ax.set_xticks(range(11)) # <--- set the ticks first
# ax.set_xticklabels([(x * 10) for x in range(11)])
# ax.set_xticks(range(5))
# ax.set_xticklabels([(x/4) for x in range(11)])
# for i, (net, time_list) in enumerate(opt_time.items()):    
# my_plotters.cdf_plt(opt_time, "ONSET Optimization Time", filename)



# plt.legend(title=None)
# ax2 = sns.ecdfplot(data=data, x="Congestion", hue="Strategy", linewidth=3, palette=palette)
# Set up the initial legend.                    
# my_legend = plt.legend(legend_key, bbox_to_anchor=(-0.2, 2), loc='upper left', borderaxespad=0, ncol=2, frameon=False)
# my_legend = plt.legend(legend_key,  bbox_to_anchor=(-0.2, 2), loc='upper left', borderaxespad=0, ncol=2, frameon=False)  
# handles, labels = my_legend.get_legend_handles_labels()
# plt.axvline(1, linestyle='--', color='black', linewidth=2)

# plt.tight_layout()

# break
# with open(figure_name + "_stats.txt", 'w') as fob:
#     fob.write("\t\t\t,Network,\tLoss Events,\ttotal\n")
#     fob.write(f"{routing},\t{network},\t{len(baseline_df[baseline_df['Loss'] > 0])},\t{len(baseline_df)}\n")
#     fob.write(f"{routing}+ONSET,\t{network},\t{len(onset_df[onset_df['Loss'] > 0])},\t{len(onset_df)}")
# print(f"saved: {figure_name}_stats.txt")
    # print( data[["Network", "Strategy", "Routing", "Congestion"]].sort_values(by="Strategy") )
