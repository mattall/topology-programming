from os import path
from matplotlib import lines
import numpy as np
from numpy import linspace, sort, absolute, arange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (
    PercentFormatter,
    FormatStrFormatter,
    AutoMinorLocator,
    LogLocator,
    NullFormatter,
)
from matplotlib.font_manager import FontProperties
from networkx import draw
import pandas as pd
import seaborn as sns
from onset.constants import PLOT_DIR

from onset.constants import PLT_BASE, PLT_HIGHT
from onset.utilities.sysUtils import save_raw_data

plt.rcParams.update(
    {
        "figure.constrained_layout.use": True,
        "font.size": 26,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 2,
        "xtick.major.size": 12,
        "ytick.major.size": 12,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.minor.size": 6,
        "ytick.minor.size": 6,
        "xtick.minor.width": 2,
        "ytick.minor.width": 2,
        "lines.linewidth": 2,
    }
)

def draw_graph(
    G, name, node_color="white", with_labels=True, edge_color="black"
):
    """Draws graph object, placing nodes according to Longitude and Latitude attributes.

    Args:
        G (Graph): [description]
        node_color (str, optional): [description]. Defaults to 'blue'.
    """
    pos = {}
    for node in G.nodes():
        pos[node] = (G.nodes()[node]["Longitude"], G.nodes()[node]["Latitude"])

    # pprint(G.nodes())
    # pos = spring_layout(G)
    # pprint(pos)
    draw(
        G,
        pos,
        node_color=node_color,
        with_labels=with_labels,
        edge_color=edge_color,
    )
    plt.savefig(name)
    plt.close()


def congestion_heatmap(data: dict, name: str):
    ser = pd.Series(
        list(data.values()), index=pd.MultiIndex.from_tuples(data.keys())
    )
    df = ser.unstack().fillna(0)
    my_congestion_heatmap = sns.heatmap(df, vmax=1, cmap="YlGnBu")

    # fig, ax = plt.subplots()
    # im = ax.imshow(data)
    # source = [d[0] for d in data.keys()]
    # target = [d[1] for d in data.keys()]
    # # We want to show all ticks...
    # ax.set_xticks(np.arange(len(data)))
    # ax.set_yticks(np.arange(len(data)))
    # # ... and label them with the respective list entries
    # ax.set_xticklabels(source)
    # ax.set_yticklabels(target)

    # # Rotate the tick labels and set their alignment.
    # # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    # #         rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # for i in range(len(source)):
    #     for j in range(len(target)):
    #         text = ax.text(j, i, data[i, j],
    #                     ha="center", va="center", color="w")

    # ax.set_title("Edge Congestion")
    # fig.tight_layout()
    plt.savefig(name + ".pdf")
    plt.savefig(name + ".png")


def plot_timeseries(
    X,
    Y,
    num_lines,
    xlabel="X",
    ylabel="Y",
    name="untitled",
    series_labels=[],
    save_legend=True,
    pass_X_direct=False,
    log_scale=False,
    ylim=None,
):
    if num_lines > 1 and num_lines != len(series_labels):
        raise (Exception, "Error, each line must have a label!")

    if num_lines == 1:
        if len(Y) != len(X):
            Y = Y[0]
            if len(Y) != len(X):
                raise ("BAD ARGUMENTS, X:{} Y:{}".format(X, Y))

    font = {"size": 18}
    mpl.rc("font", **font)

    # fig, ax = plt.subplots(figsize=(6, 8))
    fig, ax = plt.subplots()
    plt.tick_params(axis="both", width=4, length=6)
    plt.tick_params(axis="both", which="major", width=4, length=6)
    plt.tick_params(axis="both", which="minor", width=2, length=6)

    marks = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "8",
        "s",
        "p",
        "*",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]
    linestyle = ["-", "--", "-.", "-", "--", "-.", "-", "--", "-."]
    if num_lines > 1:
        # if True:
        for line in range(num_lines):
            if pass_X_direct:
                plt.plot(
                    X[line],
                    Y[line],
                    marker=marks[line],
                    label=series_labels[line],
                    linestyle=linestyle[line],
                    markersize=8,
                )
            else:
                plt.plot(
                    X[: len(Y[line])],
                    Y[line],
                    marker=marks[line],
                    label=series_labels[line] if series_labels else "",
                    linestyle=linestyle[line],
                    markersize=8,
                )

    else:
        plt.plot(X, Y, marker=marks[0])

    if ylim:
        plt.ylim(ylim[0], ylim[1])

    if log_scale:
        plt.yscale("symlog")
        # minor_locator = AutoMinorLocator(10)
        # ax.yaxis.set_minor_locator(minor_locator)
        y_major = LogLocator(base=10.0, numticks=5)
        ax.yaxis.set_major_locator(y_major)
        y_minor = LogLocator(
            base=10.0, subs=arange(1.0, 10.0) * 0.1, numticks=10
        )
        ax.yaxis.set_minor_locator(y_minor)
        ax.yaxis.set_minor_formatter(NullFormatter())

        # plt.grid(True, which="both", linestyle='--')

    if num_lines > 1 and save_legend == True:
        # plt.legend()
        legend = plt.legend(
            loc=3, framealpha=1, frameon=True, facecolor="white", ncol=3
        )

        def export_legend(legend, filename="{}_legend.png".format(name)):
            fig = legend.figure
            fig.canvas.draw()
            bbox = legend.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted()
            )
            fig.savefig(filename, bbox_inches=bbox)
            fig.savefig(filename.replace("png", "pdf"), bbox_inches=bbox)
            print("Saving legend as: {}".format(filename))

        export_legend(legend)
        legend.remove()

    # elif num_lines > 1 and save_legend == False:
    #     plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()

    plt.savefig(name + ".pdf")
    plt.savefig(name + ".png")
    plt.close()


def plot_reconfig_time(
    X,
    Y,
    num_lines,
    xlabel="X",
    ylabel="Y",
    name="untitled",
    series_labels=[],
    save_legend=True,
):
    if num_lines > 1 and num_lines != len(series_labels):
        raise (Exception, "Error, each line must have a label!")

    font = {"size": 22}
    mpl.rc("font", **font)
    ax = plt.subplot()

    marks = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "8",
        "s",
        "p",
        "*",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]
    linestyle = ["-", "--", "-.", "-", "--", "-.", "-", "--", "-."]
    if num_lines > 1:
        for line in range(num_lines):
            plt.plot(
                X[: len(Y[line])],
                Y[line],
                marker=marks[line],
                label=series_labels[line],
                linestyle=linestyle[line],
                markersize=8,
            )

    else:
        plt.plot(X, Y[0], marker=marks[0])

    if num_lines > 1 and save_legend == True:
        # plt.legend()
        legend = plt.legend(
            loc=3, framealpha=1, frameon=True, facecolor="white"
        )

        def export_legend(legend, filename="{}_legend.png".format(name)):
            fig = legend.figure
            fig.canvas.draw()
            bbox = legend.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted()
            )
            fig.savefig(filename, bbox_inches=bbox)
            fig.savefig(filename.replace("png", "pdf"), bbox_inches=bbox)

        export_legend(legend)
        legend.remove()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()

    plt.savefig(name + ".pdf")
    plt.savefig(name + ".png")
    plt.close()


def cdf_average_congestion(data_in: list, plot_name):
    font = {"size": 22}
    mpl.rc("font", **font)
    ax = plt.subplot()
    X = list(sort(data_in))
    Y = linspace(0, 100, len(X))
    plt.plot(X, Y)
    plt.axvline(
        color="gray",
    )
    plt.margins(0)
    ax.set_xlabel("Change in Average Link Congestion")
    ax.set_ylabel("CDF")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    plt.tight_layout()
    plt.savefig(plot_name + ".pdf")
    plt.savefig(plot_name + ".png")
    plt.clf()


def cdf_churn(data_in: list, plot_name):
    font = {"size": 22}
    mpl.rc("font", **font)
    ax = plt.subplot()

    X = list(sort(data_in))
    Y = linspace(0, 100, len(X))
    plt.plot(X, Y)
    plt.margins(0)
    ax.set_xlabel("Path Churn")
    ax.set_ylabel("CDF")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    plt.tight_layout()
    plt.savefig(plot_name + ".pdf")
    plt.savefig(plot_name + ".png")
    plt.close()
    cdf_absolute_churn(data_in, plot_name)


def cdf_absolute_churn(data_in: list, plot_name):
    font = {"size": 22}
    mpl.rc("font", **font)
    ax = plt.subplot()
    X = sort(absolute(data_in))
    Y = linspace(0, 100, len(X))
    plt.plot(X, Y)
    plt.margins(0)
    ax.set_xlabel("Absolute Path Churn")
    ax.set_ylabel("CDF")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    plt.tight_layout()
    plt.savefig(plot_name + ".pdf")
    plt.savefig(plot_name + ".png")
    plt.close()


def plot_points(X: list, Y: list, x_name, y_name, plot_name):
    font = {"size": 22}
    mpl.rc("font", **font)
    plt.axhline(y=0, color="gray")
    plt.axvline(x=0, color="gray")
    plt.scatter(X, Y)
    # plt.xlim(-10, 10)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    # plt.legend(fontsize=14)
    ax = plt.subplot()

    if "diff" in plot_name:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    else:
        if "congestion" in x_name.lower() or "diff" in x_name.lower():
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        else:
            ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))

        if "congestion" in x_name.lower() or "diff" in x_name.lower():
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

    plt.scatter(X, Y)

    plt.tight_layout()
    plt.savefig(plot_name + ".pdf")
    plt.savefig(plot_name + ".png")
    save_series(X, Y, x_name, y_name, plot_name)
    plt.close()


def save_series(X: list, Y: list, x_name, y_name, plot_name):
    with open(plot_name + ".csv", "w") as fob:
        fob.write("{},{}\n".format(x_name, y_name))
        for x, y in zip(X, Y):
            fob.write("{},{}\n".format(x, y))


def save_dictionary(d: dict, name):
    df = pd.DataFrame.from_dict(d, orient="index")
    df = df.transpose()
    df.to_csv(name + ".csv", index=False)


def congestion_multi_cdf(
    data_in, inLabel, title=None, fig_name="cdf", scatter=False, labels=None
):
    # data_in: A dictionary of lists indexed by a name.
    #           ex. {"mcf: [0,1,2]"}
    #
    # inLabel: A name for the x-axis
    # Returns: None
    # Effect: Saves CDF of data as fig_name.png
    # 	      Will create a different line in the cdf for each
    #         named list in data_in
    font = {"size": 32}
    mpl.rc("font", **font)
    marks = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "8",
        "s",
        "p",
        "*",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]
    styles = [
        "-",  # solid
        "-.",  # dash
        (0, (5, 5)),  # dashed
        (0, (5, 1)),  # desely dashed
        (0, (1, 1)),  # dotted
        # (0, ()), # solid
    ]
    widths = [2]
    # mpl.rcParams.update({'font.size': 14})
    mpl.rcParams["axes.linewidth"] = 2
    # plt.rcParams[font.family] = "Times New Roman"
    # fig, ax = plt.subplots(figsize=(10, 6))
    fig, ax = plt.subplots()
    # plot the cumulative histogram
    print(data_in.keys())
    keys = list(data_in.keys())
    total_plots = 0
    for i, key in enumerate(sorted(keys)[1:]):
        marker = marks[i % len(marks)]
        style = styles[i % len(styles)]
        width = widths[i % len(widths)]
        N = len(data_in[key])
        if N == 1:
            plt.axvline(
                data_in[key][0],
                color="gray",
            )
        else:
            X = list(np.sort(data_in[key]))
            X_ltz = [x for x in X if x < 0]
            # Y = list(np.linspace(1/float(N), 1, N))
            Y = list(np.linspace(0, 1, N))
            Y_ltz = list(np.linspace(0, 1, len(X_ltz)))
            # if len(X) < 20:
            #     plt.plot(X,Y, linewidth=4, label = key, marker=marker, linestyle=style, markersize=8)
            #     total_plots += 1
            # else:
            plt.plot(
                X,
                Y,
                linewidth=4,
                label=key,
                marker=marker,
                linestyle=style,
                markersize=8,
            )
            # plt.plot(X_ltz,Y_ltz, linewidth=4, label = key, marker=marker, linestyle=style, markersize=8)
            total_plots += 1

    # legend = plt.legend(loc=3, framealpha=1,  frameon=True, facecolor='white')
    if total_plots > 1:
        legend = plt.legend()
    # def export_legend(legend, filename="{}_legend.png".format(fig_name)):
    #     fig  = legend.figure
    #     fig.canvas.draw()
    #     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #     fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    # export_legend(legend)
    ax = plt.gca()
    # ax.legend_ = None
    # if len(data_in.keys()) > 1:
    #     plt.legend(prop=fontP)

    ax.set_xlabel(
        inLabel + "\nOriginal: {:.1f}%".format(float(data_in[0][0]) * 100)
    )
    if "diff" in fig_name:
        plt.axvline(color="gray", linewidth=3)
    else:
        plt.axvline((data_in[0][0]), color="gray", linewidth=3)

    ax.set_ylabel("CDF")
    ax.tick_params(length=8, width=2)
    ax.set_yticks(np.linspace(0, 1, num=5))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # if "difference" in fig_name:
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    # else:
    #     ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))

    # fig.set_size_inches(10,6)
    # plt.title(title + ". Total Candidate Links: {}.".format(N))
    plt.margins(0.02)
    # fig.subplots_adjust(bottom = 0.21, left = 0.20)
    # fig.subplots_adjust(left=0.137, bottom=0.19, right=0.948, top=0.939)
    fig.tight_layout(pad=0.1)

    # plt.show()
    print("saving figures: {}".format(fig_name))
    save_dictionary(data_in, fig_name)
    plt.savefig(fig_name + ".pdf")
    plt.savefig(fig_name + ".png")
    plt.close()


def congestion_multi_cdf_v2(
    data_in,
    original_vals,
    key_order,
    inLabel,
    title=None,
    fig_name="cdf",
    scatter=False,
    labels=None,
):
    # data_in: A dictionary of lists indexed by a name.
    #           ex. {"mcf: [0,1,2]"}
    #
    # inLabel: A name for the x-axis
    # Returns: None
    # Effect: Saves CDF of data as fig_name.png
    # 	      Will create a different line in the cdf for each
    #         named list in data_in
    font = {"size": 32}
    mpl.rc("font", **font)
    marks = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "8",
        "s",
        "p",
        "*",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]
    styles = [
        "-",  # solid
        "-.",  # dash
        (0, (5, 5)),  # dashed
        (0, (5, 1)),  # desely dashed
        (0, (1, 1)),  # dotted
        # (0, ()), # solid
    ]
    widths = [2]
    # mpl.rcParams.update({'font.size': 14})
    mpl.rcParams["axes.linewidth"] = 2
    # plt.rcParams[font.family] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(8, 4))
    # fig, ax = plt.subplots()
    # plot the cumulative histogram
    print(data_in.keys())
    keys = list(data_in.keys())
    total_plots = 0
    for i, key in enumerate(key_order):
        marker = marks[i % len(marks)]
        style = styles[i % len(styles)]
        width = widths[i % len(widths)]
        N = len(data_in[key])
        # if N == 1:
        #     plt.axvline(data_in[key][0], color='gray',)
        # else:
        if 1:
            X = list(np.sort(data_in[key]))
            X_ltz = [x for x in X if x < 0]
            # Y = list(np.linspace(1/float(N), 1, N))
            Y = list(np.linspace(0, 1, N))
            Y_ltz = list(np.linspace(0, 1, len(X_ltz)))
            # if len(X) < 20:
            #     plt.plot(X,Y, linewidth=4, label = key, marker=marker, linestyle=style, markersize=8)
            #     total_plots += 1
            # else:
            plt.plot(
                X,
                Y,
                linewidth=4,
                label=key,
                marker=marker,
                linestyle=style,
                markersize=8,
            )
            # plt.plot(X_ltz,Y_ltz, linewidth=4, label = key, marker=marker, linestyle=style, markersize=8)
            total_plots += 1
            plt.axvline((original_vals[key][0]), color="gray", linewidth=3)

    # legend = plt.legend(loc=3, framealpha=1,  frameon=True, facecolor='white')
    if total_plots > 1:
        legend = plt.legend()
    # def export_legend(legend, filename="{}_legend.png".format(fig_name)):
    #     fig  = legend.figure
    #     fig.canvas.draw()
    #     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #     fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    # export_legend(legend)
    ax = plt.gca()
    # ax.legend_ = None
    # if len(data_in.keys()) > 1:
    #     plt.legend(prop=fontP)

    ax.set_xlabel("Congestion")
    # if "diff" in fig_name:
    #     plt.axvline(color='gray',linewidth=3)
    # else:
    #     plt.axvline((data_in[0][0]), color='gray', linewidth=3)

    ax.set_ylabel("CDF")
    ax.tick_params(length=8, width=2)
    ax.set_yticks(np.linspace(0, 1, num=5))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # if "difference" in fig_name:
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    # else:
    #     ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))

    # fig.set_size_inches(10,6)
    # plt.title(title + ". Total Candidate Links: {}.".format(N))
    plt.margins(0.02)
    # fig.subplots_adjust(bottom = 0.21, left = 0.20)
    # fig.subplots_adjust(left=0.137, bottom=0.19, right=0.948, top=0.939)
    fig.tight_layout(pad=0.1)

    # plt.show()
    print("saving figures: {}".format(fig_name))
    save_dictionary(data_in, fig_name)
    plt.savefig(fig_name + ".pdf")
    plt.savefig(fig_name + ".png")
    plt.close()


def generic_multi_cdf(
    data_in, inLabel, title=None, fig_name="cdf", scatter=False, labels=None
):
    # data_in: A dictionary of lists indexed by a name.
    #           ex. {"mcf: [0,1,2]"}
    #
    # inLabel: A name for the x-axis
    # Returns: None
    # Effect: Saves CDF of data as fig_name.png
    # 	      Will create a different line in the cdf for each
    #         named list in data_in
    font = {"size": 22}
    mpl.rc("font", **font)
    marks = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "8",
        "s",
        "p",
        "*",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
    ]
    styles = [
        "-",  # solid
        "-.",  # dash
        (0, (5, 5)),  # dashed
        (0, (5, 1)),  # desely dashed
        (0, (1, 1)),  # dotted
        # (0, ()), # solid
    ]
    widths = [2]
    # mpl.rcParams.update({'font.size': 14})
    mpl.rcParams["axes.linewidth"] = 2
    # plt.rcParams[font.family] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(10, 6))
    # plot the cumulative histogram
    print(data_in.keys())
    keys = list(data_in.keys())
    for i, key in enumerate(sorted(keys)):
        marker = marks[i % len(marks)]
        style = styles[i % len(styles)]
        width = widths[i % len(widths)]
        N = len(data_in[key])
        if N == 1:
            plt.axvline(
                data_in[key][0],
                color="gray",
            )
        else:
            X = list(np.sort(data_in[key]))
            # Y = list(np.linspace(1/float(N), 1, N))
            Y = list(np.linspace(0, 1, N))
            if len(X) < 20:
                plt.plot(
                    X,
                    Y,
                    linewidth=4,
                    label=key,
                    marker=marker,
                    linestyle=style,
                )

            else:
                plt.plot(X, Y, linewidth=width, label=key, linestyle=style)

    # legend = plt.legend(loc=3, framealpha=1,  frameon=True, facecolor='white')
    legend = plt.legend()
    # def export_legend(legend, filename="{}_legend.png".format(fig_name)):
    #     fig  = legend.figure
    #     fig.canvas.draw()
    #     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #     fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    # export_legend(legend)
    ax = plt.gca()
    # ax.legend_ = None
    # if len(data_in.keys()) > 1:
    #     plt.legend(prop=fontP)

    ax.set_xlabel(inLabel)

    ax.set_ylabel("CDF")
    ax.tick_params(length=8, width=2)
    ax.set_yticks(np.linspace(0, 1, num=5))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    # if "difference" in fig_name:
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    # else:
    #     ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))

    # fig.set_size_inches(10,6)
    # plt.title(title + ". Total Candidate Links: {}.".format(N))
    plt.margins(0)
    # fig.subplots_adjust(bottom = 0.21, left = 0.20)
    fig.subplots_adjust(left=0.137, bottom=0.19, right=0.948, top=0.939)
    # plt.show()
    print("saving figures: {}".format(fig_name))
    save_dictionary(data_in, fig_name)
    plt.savefig(fig_name + ".pdf")
    plt.savefig(fig_name + ".png")
    plt.close()


def new_figure(scale=1):
    return plt.subplots(figsize=(scale * PLT_BASE, scale * PLT_HIGHT))


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.
    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def cdf_plt(
    distribution,
    xlabel="X",
    output_file="cdf",
    complement=False,
    clear_fig=True,
    fig=False,
    ax=False,
    label="",
):    
    if not isinstance(distribution, np.ndarray):
        distribution = np.array(distribution)        
        distribution[np.isnan(distribution)] = 0
        
    output_file = path.join(PLOT_DIR, output_file)
    s = distribution.sum()
    cdf = distribution.cumsum(0) / s
    # sort the data:
    data_sorted = np.sort(distribution)

    # calculate the proportional values of samples
    p = 1.0 * np.arange(len(distribution)) / (len(distribution) - 1)

    if fig and ax:
        pass
    else:
        fig, ax = new_figure(scale=3)

    if complement:
        ccdf = 1 - p
        ylabel = xlabel
        xlabel = "CCDF"
        X = ccdf
        Y = data_sorted
        ax.set_xlim([0, 1])
        # ax.set_ybound(lower=0)
        ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        # ax.set_yticks([0, 10, 20, 30])
        save_raw_data(X, Y, output_file, ylabel, xlabel)
    else:
        ylabel = "CDF"
        Y = cdf
        X = data_sorted
        ax.set_ylim([0, 1])
        save_raw_data(X, Y, output_file, xlabel, ylabel)

    ax.plot(X, Y, label=label)
    ax.grid()
    ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    # ax.set_yticks([0, 10, 20, 30])
    fig.savefig(output_file + ".jpg")
    fig.savefig(output_file + ".pdf")
    print(f"saved plot to {output_file}.jpg")
    if clear_fig:
        plt.clf()
        return None
    else:
        return fig, ax

