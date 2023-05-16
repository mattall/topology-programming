"""
attack_generator.py
author: MNH

This script is used to create attack traffic matrices. 

Read paths from path file.

Find k most used edges (list:k_edges), and flows for those edges {dict: int(edge_k) -> list(flows)}.

Example input file:
h1 -> h2 :
[(h1,s1), (s1,s5), (s5,s11), (s11,s2), (s2,h2)] @ 0.333333
[(h1,s1), (s1,s8), (s8,s7), (s7,s2), (s2,h2)] @ 0.333333
[(h1,s1), (s1,s8), (s8,s11), (s11,s2), (s2,h2)] @ 0.333333

About Matrix encodings for input/output files:
# matrix is encoded in single line format
# A = [ 0,0   0,1   0,2
#       1,0   1,1   1,2
#       2,0   2,1   2,2 ]
#  is written as A = [ 0,0  0,1  0,2  1,0  1,1  1,2  2,0  2,1  2,1 ]
"""

from collections import defaultdict
import json
from webbrowser import get
import numpy as np
import networkx as nx
import sys

from sys import argv
from math import inf
from os import path
from typing import Counter, DefaultDict
from onset.utilities import ZERO_INDEXED
from onset.utilities.graph import get_paths, parse_edges
from onset.utilities.tmg import rand_gravity_matrix
from utilities import SCRIPT_HOME, ZERO_INDEXED
from onset.utilities.flows import read_tm_to_tc
from onset.utilities.graph import (
    convert_paths_onset_to_json,
)
from tmgen.models import modulated_gravity_tm


def save_mtrx_to_txt(matrix, name):
    print("Saving data to: {}".format(name))
    try:
        np.savetxt(name, (matrix).astype(int)[None], fmt="%i")
    except Exception as e:
        np.savetxt(name, (matrix).astype(int), fmt="%i")


def save_attack_to_json(
    json_paths_file, n_most_used_edges, complete_mtrx, og_mtrx, name
):
    # Prepare attack data as JSON
    target_link = {}
    effective_cong = {}
    path_obj = json.load(open(json_paths_file, "r"))
    paths = path_obj["paths"]
    for id, link in enumerate(n_most_used_edges):
        link_id = "link" + str(id)

        if "s" in str(link[0]):  # Wont be true if paths are read from onset.
            src = link[0]
            dst = link[1]
        else:
            src = "s" + str(link[0])
            dst = "s" + str(link[1])

        target_link[link_id] = {"src": src, "dst": dst}

        effective_cong["[{}, {}]".format(link[0], link[1])] = 1.0

    tc = {}
    attack_tm = complete_mtrx - og_mtrx
    # this function modifies the dictionary passed in its first argument, tc.
    read_tm_to_tc(tc, og_mtrx, paths, malicious=False, rolling=-1)
    read_tm_to_tc(tc, attack_tm, paths, malicious=True, rolling=1)

    attack_json = {
        "effective cong": effective_cong,
        "target_link": target_link,
        "nlink": len(n_most_used_edges),
        "traffic_class": tc,
    }

    # Save data in JSON
    attack_json_file = name + ".json"
    with open(attack_json_file, "w") as fob:
        json.dump(attack_json, fob, indent=4)
    print("flows dumped to: {}".format(attack_json_file))


class Attacker:
    def __init__(self, network, path_file, n_hosts=-1):
        # n_hosts is required for the methods, `Attacker.make_coremelt_attack_matrix` and `Attacker.rolling_attack`
        self.network = network
        self.most_used_edge = None
        # (edge:(core_node, core_node) -> int:count) increment int each time edge is seen.
        self.edge_use = DefaultDict(int)
        # (edge:(core_node, core_node) -> list:[(host_node, host_node), ...]) append flow (host pair) for each edge
        self.edge_flows = DefaultDict(list)
        self.target_edges = []
        self.path_file = path_file
        # load paths as json format
        self.n_hosts = n_hosts
        if path_file.endswith(".json"):
            self.txt_path_file = None  # Paths derived by Yates or GML file
            self.json_path_file = path_file
            with open(self.json_path_file, "r") as fob:
                self.json_paths = json.load(fob)
        else:
            self.txt_path_file = self.path_file
            # convert paths.
            self.json_path_file = path.join(
                SCRIPT_HOME, "data", "paths", network + ".json"
            )
            self.json_paths = convert_paths_onset_to_json(
                self.txt_path_file, self.json_path_file
            )

    def set_target_link(self, u, v):
        self.most_used_edge = (u, v)

    def find_target_link_from_txt(self, n_edges=1):
        with open(self.path_file, "r") as fob:
            a, b = None, None
            for line in fob:
                if line.startswith("h"):
                    host_line = line.split(" -> ")
                    a, b = [h.strip(" \n:") for h in host_line]
                    # if ZERO_INDEXED: # zero indexed nodes
                    # a = str(int(a.strip('h')) - 1)
                    # b = str(int(b.strip('h')) - 1)
                    a = str(int(a.strip("h")))
                    b = str(int(b.strip("h")))

                if line.startswith("["):  # line contains a path
                    path, percent = line.strip().split("@")
                    path_edges = parse_edges(path)
                    for edge in path_edges:
                        self.edge_use[edge] += 1
                        self.edge_flows[edge].append((a, b))
                        if self.edge_use[edge] == max(self.edge_use.values()):
                            self.most_used_edge = edge
        self.set_n_most_used_edges(n_edges)

    def find_target_link(self, n_edges=1):
        paths = self.json_paths["paths"]
        for net_path in paths:
            a = paths[net_path]["src"]
            b = paths[net_path]["dst"]
            hops = paths[net_path]["hops"]
            for u, v in zip(hops[:], hops[1:]):
                edge = (u, v)
                self.edge_use[edge] += 1
                self.edge_flows[edge].append((a, b))
                if self.edge_use[edge] == max(self.edge_use.values()):
                    self.most_used_edge = edge

        self.set_n_most_used_edges(n_edges)

    def set_n_most_used_edges(self, n) -> list:
        n_most_used_edges = set()
        d = Counter(self.edge_use)
        for k, v in d.most_common():
            if type(k[0]) == int:
                edge = tuple(sorted([int(k[0]), int(k[1])]))
            else:
                edge = tuple(sorted([k[0], k[1]]))
            n_most_used_edges.add(edge)
            if len(n_most_used_edges) == n:
                break

        self.n_most_used_edges = sorted(list(n_most_used_edges))
        # self.n_most_used_edges = [(str(a), str(b)) for (a, b) in n_most_used_edges]

    def scale_naive_attack(self, traffic_matrix, iterations, strength, tag=""):
        # Similar to a Spiffy attack.
        # Cost sensitive attacker scales their attack traffic until a certain cost (strength)

        num_targets = len(self.n_most_used_edges)
        attack_flows = self.edge_flows[self.most_used_edge]

        # load original matrix.
        og_matrix = np.loadtxt(traffic_matrix, dtype=int)
        dim = int(np.sqrt(len(og_matrix)))

        # instance new_matrix
        new_matrix = np.zeros((iterations + 1, len(og_matrix)))
        for i in range(iterations + 1):
            new_matrix[i] = og_matrix
            for (
                target_edge_source,
                target_edge_destination,
            ) in self.n_most_used_edges:
                attack_flows = self.edge_flows[
                    (target_edge_source, target_edge_destination)
                ]
                flow_strength = strength / len(attack_flows)
                # new_matrix[i] = og_matrix
                # Assign flows for the attack.
                for source, target in attack_flows:
                    s = source.strip("s")
                    t = target.strip("s")
                    index = int((dim * (int(s) - 1)) + (int(t) - 1))
                    new_matrix[i][index] += i * flow_strength

            if num_targets == 1:  # put target link in file name.
                # save traffic matrix after assigning flows.
                attack_iteration_matrix = (
                    "{0}_link_{1}_{2}_strength_{3}".format(
                        traffic_matrix[:-4],
                        target_edge_source,
                        target_edge_destination,
                        i,
                    )
                )
            else:  # save target links IDs in separate file.
                attack_iteration_matrix = (
                    "{0}_targets_{1}_strength_{2}".format(
                        traffic_matrix[:-4], num_targets, i
                    )
                )

            # Save data in plain text
            print("Saving data to: {}".format(attack_iteration_matrix))
            np.savetxt(
                attack_iteration_matrix,
                (new_matrix[i] - og_matrix).astype(int)[None],
                fmt="%i",
            )

            # Prepare attack data as JSON
            target_link = {}
            effective_cong = {}
            path_obj = json.load(open(self.json_path_file, "r"))
            paths = path_obj["paths"]
            for id, link in enumerate(self.n_most_used_edges):
                link_id = "link" + str(id)

                # Wont be true if paths are read from onset.
                if "s" in str(link[0]):
                    src = link[0]
                    dst = link[1]
                else:
                    src = "s" + str(link[0])
                    dst = "s" + str(link[1])

                target_link[link_id] = {"src": src, "dst": dst}

                effective_cong["[{}, {}]".format(link[0], link[1])] = 1.0

            tc = {}
            attack_tm = new_matrix[i] - og_matrix
            # this function modifies the dictionary passed in its first argument, tc.
            read_tm_to_tc(tc, og_matrix, paths, malicious=False, rolling=-1)
            read_tm_to_tc(tc, attack_tm, paths, malicious=True, rolling=1)

            attack_json = {
                "effective cong": effective_cong,
                "target_link": target_link,
                "nlink": num_targets,
                "traffic_class": tc,
            }

            # Save data in JSON
            attack_json_file = attack_iteration_matrix + ".json"
            with open(attack_json_file, "w") as fob:
                json.dump(attack_json, fob, indent=4)
            print("flows dumped to: {}".format(attack_json_file))

        """
        # if num_targets == 1:
        #     result_file = "{}_{}Gbps_{}_{}.txt".format(traffic_matrix[:-4], int(strength/10**9), iterations, tag)
        
        # elif num_targets > 1: 
        #     result_file = "{}_{}x{}Gbps_{}_{}.txt".format(traffic_matrix[:-4], num_targets, int(strength/10**9), iterations, tag)

        # else:
        #     print("Error! num_targets must be >= 1. First call find_target_link")
        """
        result_file = "{}_{}x{}Gbps_{}_{}.txt".format(
            traffic_matrix[:-4],
            num_targets,
            int(strength / 10**9),
            iterations,
            tag,
        )

        print("Saving data to: {}".format(result_file))
        np.savetxt(result_file, new_matrix.astype(int), fmt="%i")

    def one_shot_sustained_attack(
        self, traffic_matrix, strength, tag="", max_link_util=inf
    ):
        # Three matrices in this attack. 1) Baseline,  2) attack, 3) attack again (sustained)
        # This attack resembles a CoreMelt attack. A target link is chosen and flooded.
        #
        # matrix is encoded in single line format
        # A = [ 0,0   0,1   0,2
        #       1,0   1,1   1,2
        #       2,0   2,1   2,2 ]
        #  is written as A = [ 0,0  0,1  0,2  1,0  1,1  1,2  2,0  2,1  2,2 ]
        length = 3
        attack_flows = self.edge_flows[self.most_used_edge]

        # load original matrix.
        og_matrix = np.loadtxt(traffic_matrix, dtype=int)
        dim = int(np.sqrt(len(og_matrix)))

        # instance new_matrix
        new_matrix = np.zeros((length, len(og_matrix)))  # 3 x matrices
        for i in range(length):
            new_matrix[i] = og_matrix
            if i == 0:
                num_targets = 0
            else:
                num_targets = len(self.n_most_used_edges)
                for (
                    target_edge_source,
                    target_edge_destination,
                ) in self.n_most_used_edges:
                    attack_flows = self.edge_flows[
                        (target_edge_source, target_edge_destination)
                    ]
                    flow_strength = min(
                        strength / len(attack_flows), max_link_util
                    )
                    # new_matrix[i] = og_matrix
                    # Assign flows for the attack.
                    for source, target in attack_flows:
                        s = source.strip("s")
                        t = target.strip("s")
                        # s, t = source, target
                        index = int((dim * (int(s) - 1)) + (int(t) - 1))
                        new_matrix[i][index] += flow_strength

            # if num_targets == 1: # put target link in file name.
            #     # save traffic matrix after assigning flows.
            #     attack_iteration_matrix="{0}_link_{1}_{2}_strength_{3}".format(traffic_matrix[:-4],
            #                                                 target_edge_source, target_edge_destination, i)
            # else: # save target links IDs in separate file.

            # attack_file = "./data/traffic/{0}_targets_{1}_iteration_{2}_strength_{3}_atk".format(traffic_matrix[:-4],
            #                                                                       num_targets, i, int(strength/10**9))
            # save_mtrx_to_txt((new_matrix[i] - og_matrix), attack_file)

            # mixed_file = "./data/traffic/{0}_targets_{1}_iteration_{2}_strength_{3}_mix".format(traffic_matrix[:-4],
            #                                                                      num_targets, i, int(strength/10**9))
            # save_mtrx_to_txt((new_matrix[i]), mixed_file)

            # json_file = "./data/traffic/{0}_targets_{1}_iteration_{2}_strength_{3}".format(traffic_matrix[:-4],
            #                                                                 num_targets, i, int(strength/10**9))

            json_file = "{0}_targets_{1}_iteration_{2}_strength_{3}".format(
                traffic_matrix[:-4], num_targets, i, int(strength / 10**9)
            )

            save_attack_to_json(
                self.json_path_file,
                self.n_most_used_edges,
                new_matrix[i],
                og_matrix,
                json_file,
            )
        """ DEAD CODE. SAVE ALL NAMES AS IF >1 Link case. 
        if num_targets == 1:
            result_file = "{}_{}Gbps_{}_{}.txt".format(traffic_matrix[:-4], int(strength/10**9), length, tag)
        
        elif num_targets > 1: 
            result_file = "{}_{}x{}Gbps_{}_{}.txt".format(traffic_matrix[:-4], num_targets, int(strength/10**9), length, tag)

        else:
            print("Error! num_targets must be >= 1. First call find_target_link")
        """
        result_file = "{}_{}x{}Gbps_{}_{}.txt".format(
            traffic_matrix[:-4],
            num_targets,
            int(strength / 10**9),
            length,
            tag,
        )
        print("Saving data to: {}".format(result_file))
        np.savetxt(result_file, new_matrix.astype(int), fmt="%i")

    def make_coremelt_attack_matrix(
        self,
        n_hosts: int,
        n_targets: int,
        strength: int,
        max_link_util=inf,
        target_edges=None,
        save_matrix=True,
    ):
        # Random linkflood attack.
        # Targets chosen at random.
        dim = n_hosts
        edge_list = list(self.edge_flows.keys())
        if target_edges == None:
            # n_targets = len(self.n_most_used_edges)
            target_edges_i = np.random.choice(
                range(len(self.edge_flows)), n_targets, replace=False
            )

            # attack_flows = self.edge_flows[target_edges]
            # load original matrix.
            target_edges = [edge_list[i] for i in target_edges_i]
            print("target_edges: ", target_edges)

        if target_edges is not None and len(target_edges) < n_targets:
            missing_targets = n_targets - len(target_edges)
            # existing_target_indices = [edge_list.index(target_edge) in target_edges]
            possible_targets = edge_list[:]
            for eti in target_edges:
                possible_targets.remove(eti)

            rand_targets = np.random.choice(
                range(len(possible_targets)), missing_targets, replace=False
            )
            target_edges.extend(possible_targets[i] for i in rand_targets)

        # instance new_matrix
        new_matrix = np.zeros(dim**2)
        for target_edge_source, target_edge_destination in target_edges:
            attack_flows = self.edge_flows[
                (target_edge_source, target_edge_destination)
            ]
            flow_strength = min(strength / len(attack_flows), max_link_util)
            # new_matrix[i] = og_matrix
            # Assign flows for the attack.

            for source, target in attack_flows:
                s = source.strip("s")
                t = target.strip("s")
                # s, t = source, target
                index = int((dim * (int(s) - 1)) + (int(t) - 1))
                new_matrix[index] += flow_strength

        if save_matrix:
            attack_file = "./data/traffic/{0}_targets_{1}_strength_{2}".format(
                self.network, n_targets, int(strength / 10**9)
            )
            save_mtrx_to_txt(new_matrix, attack_file)

        return (new_matrix.astype(int), target_edges)

    def get_edges(self):
        G = nx.Graph()
        G.add_edges_from(self.edge_flows.keys())
        return list(G.edges())

    def make_crossfire_attack_matrix(
        self,
        n_hosts,
        strength,
        target_node=None,
        max_link_util=inf,
        save_matrix=True,
    ) -> tuple:
        """Crossfire attack targets a region of the network.
        Attack chooses a node from the network and floods the incident edges.

        Args:
            n_hosts (int): number of network hosts
            strength (int): bits per second (bps) of attack volume per link flooded
            max_link_util (int, optional): limits attack volume per attack flow. Defaults to inf.

        Returns:
            tuple (list, list): (traffic matrix, links targeted)
        """
        # create graph from edge data.
        G = nx.Graph()
        G.add_edges_from(self.edge_flows.keys())
        if target_node == None:
            center_edges = nx.center(G)
            arg_min = lambda l: l.index(min(l))
            center_degrees = [nx.degree(G, ce) for ce in center_edges]
            target_node = center_edges[arg_min(center_degrees)]

        target_edges = [(target_node, node) for node in G[target_node]]
        # target_edges_i = np.random.choice(range(len(self.edge_flows)), n_targets, replace=False)
        # attack_flows = self.edge_flows[target_edges]
        # load original matrix.
        dim = n_hosts

        # instance new_matrix
        new_matrix = np.zeros(dim**2)
        n_targets = len(target_edges)
        for target_edge_source, target_edge_destination in target_edges:
            attack_flows = self.edge_flows[
                (target_edge_source, target_edge_destination)
            ]
            flow_strength = min(strength / len(attack_flows), max_link_util)
            # new_matrix[i] = og_matrix
            # Assign flows for the attack.
            for source, target in attack_flows:
                s = source.strip("s")
                t = target.strip("s")
                # s, t = source, target
                index = int((dim * (int(s) - 1)) + (int(t) - 1))
                new_matrix[index] += flow_strength

        attack_file = (
            "data/traffic/{0}_crossfire_targets_{1}_strength_{2}".format(
                self.network, n_targets, int(strength / 10**9)
            )
        )
        if save_matrix:
            save_mtrx_to_txt(new_matrix, attack_file)
        return new_matrix.astype(int), target_edges

    def coremelt_rolling_attack(
        self, n_hosts: int, n_rounds: int, attack_period: int, calm_period: int
    ):
        """Generates series of traffic matrices for Coremelt attack.
            Series includes background (non-attack) traffic.

        Args:
            n_hosts (int): number of hosts in the network
            n_rounds (int): number of traffic matrices (rounds) to generate
            attack_period (int): number of traffic matrices per attack
            calm_period (int): number of traffic matrices between attacks

        Effect:
            Saves a file with the series of traffic matrices as :
                "{self.network}_rolling_attack_{n_rounds}_round.txt"
        """

        avg_demand = 10**6
        epochs = n_rounds * (attack_period + calm_period)
        f = 1 / 24 / 60
        attack_lower_bound = 100 * 10**9
        attack_upper_bound = 200 * 10**9

        # tm = modulated_gravity_tm(11, 30, 100*10^9, diurnal_freq=f)
        tm = modulated_gravity_tm(11, epochs, avg_demand, diurnal_freq=f)
        tm = tm.matrix.reshape(tm.num_epochs(), tm.num_nodes() ** 2).astype(
            int
        )

        attack_matrices = []
        for _ in range(n_rounds):
            n_targets = np.random.randint(1, 2)
            # strength = np.random.randint(attack_lower_bound, attack_upper_bound + 1)
            strength = attack_upper_bound
            m = self.make_coremelt_attack_matrix(
                self.n_hosts, n_targets, strength
            )
            attack_matrices.append(m)

        round = 0
        this_epoch = calm_period
        for round in range(n_rounds):
            for _ in range(attack_period):
                tm[this_epoch] += attack_matrices[round]
                this_epoch += 1
            this_epoch += calm_period

        result_file = "./data/traffic/{}_rolling_attack_{}_round.txt".format(
            self.network, n_rounds
        )
        np.savetxt(result_file, tm.astype(int), fmt="%i")

    # def mixed_rolling_attacks(self, n_rounds, attack_period, calm_period):
    def mixed_rolling_attacks(self):
        """Generates a series of traffic matrices for a coremelt attack rolling attack."""
        # epochs = n_rounds * (attack_period + calm_period)
        epochs = 12 * 60
        f = 1 / 24 / 60 / 12
        attack_lower_bound = 150 * 10**9
        attack_upper_bound = 200 * 10**9

        spiffy_start = 12 * 5
        spiffy_end = 12 * 10

        crossfire_start = 12 * 15
        crossfire_end = 12 * 20

        coremelt_1_start = 12 * 25
        coremelt_1_end = 12 * 30

        coremelt_2_start = 12 * 35
        coremelt_2_end = 12 * 40

        coremelt_3_start = 12 * 45
        coremelt_3_end = 12 * 50

        coremelt_4_start = 12 * 51
        coremelt_4_end = 12 * 55

        coremelt_5_start = 12 * 55 + 2
        coremelt_5_end = 12 * 60

        spiffy_tm = None
        spiffy_links = None

        cf_tm = None
        cf_links = None

        cm1_tm = None
        cm1_links = None

        cm2_tm = None
        cm2_links = None

        cm3_tm = None
        cm3_links = None

        cm4_tm = None
        cm4_links = None

        cm5_tm = None
        cm5_links = None

        # tm = modulated_gravity_tm(11, 30, 100*10^9, diurnal_freq=f)
        epochs = 12 * 60
        freq = 1 / 24 / 60 / 12  # an epoch is 5 seconds, 1/12 minutes.
        avg_demand = 10**6
        tm = modulated_gravity_tm(
            self.n_hosts, 12 * 60, avg_demand, diurnal_freq=freq, pm_ratio=1.05
        )
        tm = tm.matrix.reshape(tm.num_epochs(), tm.num_nodes() ** 2).astype(
            int
        )
        s_time = lambda i: "{}m:{}s".format(i // 60, i % 60)
        attack_metadata_file = path.join(
            SCRIPT_HOME,
            "data",
            "traffic",
            self.network + "_" + "rolling_attack.csv",
        )
        with open(attack_metadata_file, "w") as fob:
            fob.write("epoch, time, attack, str, links\n")
            for e in range(epochs):
                fob.write("{},\t{},\t".format(e, s_time(e * 5)))

                if e < spiffy_start:
                    fob.write("None,\t0,\t[]\n")

                elif e < spiffy_end:
                    attack_iter = e - spiffy_start + 1
                    attack_period = spiffy_end - spiffy_start
                    str = attack_iter * attack_upper_bound / attack_period
                    if spiffy_tm is None:
                        (
                            spiffy_tm,
                            spiffy_links,
                        ) = self.make_coremelt_attack_matrix(
                            self.n_hosts, 1, str
                        )

                    atk_tm = spiffy_tm * attack_iter
                    tm[e] += atk_tm

                    fob.write("spiffy,\t{},\t{}\n".format(str, spiffy_links))

                elif e < crossfire_start:
                    fob.write("None,\t0,\t[]\n")

                elif e < crossfire_end:
                    if cf_tm is None:
                        cf_tm, cf_links = self.make_crossfire_attack_matrix(
                            self.n_hosts, attack_lower_bound
                        )

                    tm[e] += cf_tm

                    fob.write(
                        "crossfire,\t{},\t{}\n".format(
                            attack_lower_bound, cf_links
                        )
                    )

                elif e < coremelt_1_start:
                    fob.write("None,\t0,\t[]\n")

                elif e < coremelt_1_end:
                    if cm1_tm is None:
                        cm1_tm, cm1_links = self.make_coremelt_attack_matrix(
                            self.n_hosts, 2, attack_upper_bound
                        )

                    tm[e] += cm1_tm
                    fob.write(
                        "coremelt,\t{},\t{}\n".format(
                            attack_upper_bound, cm1_links
                        )
                    )

                elif e < coremelt_2_start:
                    fob.write("None,\t0,\t[]\n")

                elif e < coremelt_2_end:
                    if cm2_tm is None:
                        cm2_tm, cm2_links = self.make_coremelt_attack_matrix(
                            self.n_hosts, 3, attack_upper_bound
                        )

                    tm[e] += cm2_tm
                    fob.write(
                        "coremelt,\t{},\t{}\n".format(
                            attack_upper_bound, cm2_links
                        )
                    )

                elif e < coremelt_3_start:
                    fob.write("None,\t0,\t[]\n")

                elif e < coremelt_3_end:
                    if cm3_tm is None:
                        cm3_tm, cm3_links = self.make_coremelt_attack_matrix(
                            self.n_hosts, 2, attack_upper_bound
                        )

                    tm[e] += cm3_tm
                    fob.write(
                        "coremelt,\t{},\t{}\n".format(
                            attack_upper_bound, cm3_links
                        )
                    )

                elif e < coremelt_4_start:
                    fob.write("None,\t0,\t[]\n")

                elif e < coremelt_4_end:
                    if cm4_tm is None:
                        cm4_tm, cm4_links = self.make_coremelt_attack_matrix(
                            self.n_hosts, 4, attack_lower_bound
                        )

                    tm[e] += cm4_tm
                    fob.write(
                        "coremelt,\t{},\t{}\n".format(
                            attack_lower_bound, cm4_links
                        )
                    )

                elif e < coremelt_5_start:
                    fob.write("None,\t0,\t[]\n")

                elif e < coremelt_5_end:
                    if cm5_tm is None:
                        cm5_tm, cm5_links = self.make_coremelt_attack_matrix(
                            self.n_hosts, 3, attack_lower_bound
                        )

                    tm[e] += cm5_tm
                    fob.write(
                        "coremelt,\t{},\t{}\n".format(
                            attack_lower_bound, cm5_links
                        )
                    )

                else:
                    fob.write("None,\t0,\t[]\n")

        save_mtrx_to_txt(
            tm,
            "./data/traffic/rolling_attack/{}-rolling-mixed-type-attack.txt".format(
                self.network
            ),
        )

        # attack_matrices = []
        # for _ in range(n_rounds):
        #     n_targets = np.random./randint(1, 2)
        #     # strength = np.random.randint(attack_lower_bound, attack_upper_bound + 1)
        #     strength = attack_upper_bound
        #     m = self.make_coremelt_attack_matrix(self.n_hosts, n_targets, strength)
        #     attack_matrices.append(m)

        # round = 0
        # this_epoch = calm_period
        # for round in range(n_rounds):
        #     for _ in range(attack_period):
        #         tm[this_epoch] += attack_matrices[round]
        #         this_epoch += 1
        #     this_epoch += calm_period

        # result_file = "{}_rolling_attack_{}_round.txt".format(self.network, n_rounds)
        # np.savetxt(result_file, tm.astype(int), fmt='%i')


if __name__ == "__main__":
    # networks = ["fishnet", "ANS", "bellCanada", "surfNet", "sprint"]

    try:
        network = argv[1]
        topology_file = argv[2]
        traffic_str = argv[3]
        n_targets = int(argv[4])

    except:
        network = "ANS"
        topology_file = "data/graphs/gml/ANS.gml"
        traffic_str = "200E9"
        n_targets = 2
        print(
            "USAGE: python {0} network topology_file volume number_targets\n\nExample: python {0} ANS data/graphs/gml/ANS.gml 200E9 1".format(
                argv[0]
            )
        )
        # exit()
        # netwrok = 'zayo'
        # topology_file = 'data/topologies/gml/zayo.gml'
        # traffic_str = '200E9'
        # traffic_int = int(float(traffic_str))

    traffic_int = int(float(traffic_str))
    # benign_traffic = 100E9
    benign_traffic = 0
    attack_traffic = traffic_int
    path_file = "data/paths/" + network + ".json"
    path_f, N_HOSTS = get_paths(topology_file, path_file)
    print(f"Graph has {N_HOSTS} hosts.")
    if path_f != path_file:
        print(
            f"Graph was not connected. Paths written to {path_f} instead of {path_file}"
        )
        path_file = path_f

    # test_paths = SCRIPT_HOME + "/data/results/Abilene_baseline_circuits_5__-ecmp/Abilene_1-1/paths/ecmp_0"
    # test_paths = SCRIPT_HOME + "/data/archive/2022-08-09-Azure_Flash_crowd/results/azure_baseline_2.00e+11_0__-ecmp/azure_1-433/paths/ecmp_0"
    # N_HOSTS=113
    # N_HOSTS = 11

    # test_paths = "/home/mhall/OLTE/data/results/Comcast_1-1-100/paths/ecmp_0"

    attacker = Attacker(network, path_file, n_hosts=N_HOSTS)
    attacker.find_target_link()
    base_matrix = rand_gravity_matrix(
        N_HOSTS,
        1,
        benign_traffic,
        SCRIPT_HOME
        + "/data/traffic/{}_base_tm_{:.2e}_gravity.txt".format(
            network, benign_traffic
        ),
    )
    edges = attacker.get_edges()
    assert len(attacker.edge_flows.keys()) == 2 * len(
        edges
    ), "Error number of edges ill defined"
    n_edges = len(edges)
    continuous_series = np.tile(
        base_matrix, (2 * n_edges, 1)
    )  # used to evaluate ECMP and MCF without ONSET.
    meta_str = ""
    for i in range(n_edges * 2):
        if i % 2 == 0:
            target_edge = edges[i // 2]
        else:
            target_edge = tuple(reversed(edges[i // 2]))
        attack_matrix, attacked_edges = attacker.make_coremelt_attack_matrix(
            N_HOSTS,
            n_targets,
            attack_traffic,
            100e9,
            [target_edge],
            save_matrix=False,
        )
        if len(attacked_edges) > 1:
            meta_str += "{};{}\n".format(
                i + 1, str(target_edge).replace(" ", "")
            )
        else:
            target_edges_str = (
                str(target_edge)
                .replace(" ", "")
                .replace("]", "")
                .replace("[", "")
            )
            meta_str += f"{i+1},{target_edges_str}\n"

        continuous_series[i] += attack_matrix

        if i % 2 != 0 and n_targets == 1:
            this_tm = continuous_series[i].reshape(N_HOSTS, N_HOSTS)
            prev_tm = continuous_series[i - 1].reshape(N_HOSTS, N_HOSTS)
            assert np.all(this_tm.transpose() - prev_tm == 0)

    outfile = f"{SCRIPT_HOME}/data/traffic/{network}_coremelt_links-{n_targets}_volume-{attack_traffic:.2e}.txt"
    print("Saving traffic matrix to: {}".format(outfile))
    np.savetxt(outfile, continuous_series.astype(int), fmt="%i")
    with open(
        "./data/traffic/{}_flash_crowd_meta.csv".format(network), "w"
    ) as fob:
        fob.write(meta_str.replace("s", ""))

    if 0:  # Experiment to scale traffic on a heavily shared link.
        # networks=["sprint"]
        networks = ["sprint", "ANS", "CRL", "bellCanada", "surfNet"]
        for network in networks:
            test_paths = (
                SCRIPT_HOME
                + "/data/results/{0}_add_circuit_heuristic_10/__0/paths/ecmp_0".format(
                    network
                )
            )
            test_matrix = SCRIPT_HOME + "/data/traffic/{}.txt".format(network)
            attacker = Attacker(test_paths)
            attacker.find_target_link()
            attacker.scale_naive_attack(test_matrix, 20, 10 * 10**9)

    if 0:  # Multitarget attack
        # networks=["ANS"]
        # networks = ["sprint", "ANS", "CRL", "bellCanada", "surfNet"]
        networks = ["grid_3", "linear_10", "whisker_3_2"]
        for network in networks:
            test_paths = (
                SCRIPT_HOME
                + "/data/results/{0}_add_circuit_heuristic_10/__0/paths/ecmp_0".format(
                    network
                )
            )
            test_matrix = SCRIPT_HOME + "/data/traffic/{}.txt".format(network)
            attacker = Attacker(network, test_paths)
            attacker.find_target_link(n_edges=10)
            attacker.scale_naive_attack(test_matrix, 10, 10 * 10**9)

    if 0:  # Experiment to scale traffic on Ripple's target edges.
        test_paths = (
            SCRIPT_HOME
            + "/data/results/bellCanada_add_circuit_heuristic_10/__0/paths/ecmp_0"
        )
        test_matrix = SCRIPT_HOME + "/data/traffic/bellCanada.txt"
        attacker = Attacker(test_paths)
        attacker.find_target_link()
        attacker.set_target_link("36", "41")
        attacker.scale_naive_attack(
            test_matrix, 20, 10 * 10**9, "target_sw35_sw40"
        )
        attacker.set_target_link("46", "35")
        attacker.scale_naive_attack(
            test_matrix, 20, 10 * 10**9, "target_sw45_sw34"
        )

    if 0:
        test_paths = SCRIPT_HOME + "/data/paths/sprint.json"
        attacker = Attacker("sprint", test_paths, n_hosts=11)
        attacker.find_target_link()
        attacker.make_crossfire_attack_matrix(11, 10)
        pass
        # attacker.rolling_attack(11, 3, 5, 5)

    if 0:  # ROLLING ATTACK
        # test_paths = SCRIPT_HOME + "/data/paths/sprint.json"
        # attacker = Attacker("sprint", test_paths, n_hosts=11)

        # network = "ANS"
        # for network in ["ANS", "CRL", "bellCanada", "surfNet"]:
        for network in ["bellCanada", "surfNet"]:
            test_paths = SCRIPT_HOME + "/data/paths/" + network + ".json"
            graph_path = SCRIPT_HOME + "/data/graphs/gml/" + network + ".gml"
            G = nx.read_gml(graph_path)
            N_HOSTS = len(G.nodes())
            attacker = Attacker(network, test_paths, n_hosts=N_HOSTS)
            attacker.find_target_link()
            attacker.mixed_rolling_attacks()

    if 0:  # Coremelt One Link, Both directions Link!
        from onset.utilities.tmg import rand_gravity_matrix

        network = "Abilene"
        # benign_traffic = 100E9
        benign_traffic = 0
        attack_traffic = 200e9
        test_paths = (
            SCRIPT_HOME
            + "/data/results/Abilene_baseline_circuits_5__-ecmp/Abilene_1-1/paths/ecmp_0"
        )
        N_HOSTS = 11

        attacker = Attacker(network, test_paths, n_hosts=N_HOSTS)
        attacker.find_target_link()
        edges = attacker.get_edges()
        base_matrix = np.zeros((1, N_HOSTS**2))
        n_edges = len(attacker.edge_flows.keys())
        continuous_series = np.tile(
            base_matrix, (2, 1)
        )  # used to evaluate ECMP and MCF without ONSET.
        target_edge = attacker.most_used_edge

        attack_matrix1, attacked_edges = attacker.make_coremelt_attack_matrix(
            N_HOSTS, 1, attack_traffic, 100e9, [target_edge], save_matrix=False
        )
        continuous_series[0] += attack_matrix1

        attack_matrix2, attacked_edges = attacker.make_coremelt_attack_matrix(
            N_HOSTS,
            1,
            attack_traffic,
            100e9,
            [list(reversed(target_edge))],
            save_matrix=False,
        )
        continuous_series[1] += attack_matrix2

        outfile = (
            SCRIPT_HOME
            + "/data/traffic/{}_coremelt_single_link_{:.2e}.txt".format(
                network, attack_traffic
            )
        )
        # print("Saving traffic matrix to: {}".format(outfile))
        np.savetxt(outfile, continuous_series.astype(int), fmt="%i")

    if 0:  # Coremelt Every Link Version2!
        from onset.utilities.tmg import rand_gravity_matrix

        network = "Comcast"
        # benign_traffic = 100E9
        benign_traffic = 0
        attack_traffic = 400e9
        N_HOSTS = 149

        # test_paths = SCRIPT_HOME + "/data/results/Abilene_baseline_circuits_5__-ecmp/Abilene_1-1/paths/ecmp_0"
        # test_paths = SCRIPT_HOME + "/data/archive/2022-08-09-Azure_Flash_crowd/results/azure_baseline_2.00e+11_0__-ecmp/azure_1-433/paths/ecmp_0"
        # N_HOSTS=113
        # N_HOSTS = 11

        # test_paths = "/home/mhall/OLTE/data/results/Comcast_1-1-100/paths/ecmp_0"

        attacker = Attacker(network, test_paths, n_hosts=N_HOSTS)
        attacker.find_target_link()
        base_matrix = rand_gravity_matrix(
            N_HOSTS,
            1,
            benign_traffic,
            SCRIPT_HOME
            + "/data/traffic/{}_base_tm_{:.2e}_gravity.txt".format(
                network, benign_traffic
            ),
        )
        edges = attacker.get_edges()
        assert len(attacker.edge_flows.keys()) == 2 * len(
            edges
        ), "Error number of edges ill defined"
        n_edges = len(edges)
        continuous_series = np.tile(
            base_matrix, (2 * n_edges, 1)
        )  # used to evaluate ECMP and MCF without ONSET.
        meta_str = ""
        for i in range(n_edges * 2):
            if i % 2 == 0:
                target_edge = edges[i // 2]
            else:
                target_edge = tuple(reversed(edges[i // 2]))
            (
                attack_matrix,
                attacked_edges,
            ) = attacker.make_coremelt_attack_matrix(
                N_HOSTS,
                1,
                attack_traffic,
                100e9,
                [target_edge],
                save_matrix=False,
            )
            meta_str += "{};{}\n".format(
                i + 1, str(target_edge).replace(" ", "")
            )

            continuous_series[i] += attack_matrix

            if i % 2 != 0:
                this_tm = continuous_series[i].reshape(N_HOSTS, N_HOSTS)
                prev_tm = continuous_series[i - 1].reshape(N_HOSTS, N_HOSTS)
                assert np.all(this_tm.transpose() - prev_tm == 0)

        outfile = (
            SCRIPT_HOME
            + "/data/traffic/{}_coremelt_every_link_{:.2e}.txt".format(
                network, attack_traffic
            )
        )
        print("Saving traffic matrix to: {}".format(outfile))
        np.savetxt(outfile, continuous_series.astype(int), fmt="%i")
        with open(
            "./data/traffic/{}_flash_crowd_meta.csv".format(network), "w"
        ) as fob:
            fob.write(meta_str)

    if 0:  # Coremelt Every Link!
        from onset.utilities.tmg import rand_gravity_matrix

        network = "Azure"
        # benign_traffic = 100E9
        benign_traffic = 0
        attack_traffic = 200e9
        # test_paths = SCRIPT_HOME + "/data/results/Abilene_baseline_circuits_5__-ecmp/azure_1-1/paths/ecmp_0"

        test_paths = (
            SCRIPT_HOME
            + "/data/archive/2022-08-09-Azure_Flash_crowd/results/azure_baseline_2.00e+11_0__-ecmp/azure_1-433/paths/ecmp_0"
        )
        N_HOSTS = 113

        attacker = Attacker(network, test_paths, n_hosts=N_HOSTS)
        attacker.find_target_link()
        base_matrix = rand_gravity_matrix(
            N_HOSTS,
            1,
            benign_traffic,
            SCRIPT_HOME
            + "/data/traffic/{}_base_tm_{:.2e}_gravity.txt".format(
                network, benign_traffic
            ),
        )
        n_edges = len(attacker.edge_flows.keys())
        continuous_series = np.tile(
            base_matrix, (n_edges + 1, 1)
        )  # used to evaluate ECMP and MCF without ONSET.
        for i, edge in enumerate(attacker.edge_flows.keys()):
            print("{}\t{}".format(i + 1, edge))
            (
                attack_matrix,
                attacked_edges,
            ) = attacker.make_coremelt_attack_matrix(
                N_HOSTS, 1, attack_traffic, 100e9, [edge], save_matrix=False
            )
            continuous_series[i + 1] += attack_matrix

        outfile = (
            SCRIPT_HOME
            + "/data/traffic/{}_coremelt_every_link_{:.2e}.txt".format(
                network, attack_traffic
            )
        )
        # print("Saving traffic matrix to: {}".format(outfile))
        np.savetxt(outfile, continuous_series.astype(int), fmt="%i")

    if 0:  # Crossfire Every Node!
        from onset.utilities.tmg import rand_gravity_matrix

        network = "sprint"
        test_paths = SCRIPT_HOME + "/data/paths/" + network + ".json"
        graph_path = SCRIPT_HOME + "/data/graphs/gml/" + network + ".gml"
        G = nx.read_gml(graph_path)
        N_HOSTS = len(G.nodes())

        ATTACK_STRENGTH_STRING = "200E9"
        ATTACK_STRENGTH = int(float(ATTACK_STRENGTH_STRING))

        BACKGROUND_VOLUME_STRING = "100E9"
        BACKGROUND_VOLUME = int(float(BACKGROUND_VOLUME_STRING))

        MAX_LINK_UTIL = 100e9

        attacker = Attacker(network, test_paths, n_hosts=N_HOSTS)
        attacker.find_target_link()

        outfile = (
            SCRIPT_HOME
            + "/data/traffic/"
            + network
            + "_base_tm_"
            + ATTACK_STRENGTH_STRING
            + "_gravity.txt"
        )
        base_matrix = rand_gravity_matrix(
            N_HOSTS, 1, BACKGROUND_VOLUME, outfile
        )

        G = nx.Graph()
        G.add_edges_from(attacker.edge_flows.keys())
        node_list = ["s" + str(i + 1) for i in range(len(G.nodes()))]
        n_nodes = len(node_list)
        continuous_series = np.tile(
            base_matrix, (n_nodes + 1, 1)
        )  # used to evaluate ECMP and MCF without ONSET.
        for i, node_i in enumerate(node_list):
            print(i + 1, node_i, end=" ")
            (
                attack_matrix,
                attacked_edges,
            ) = attacker.make_crossfire_attack_matrix(
                N_HOSTS,
                ATTACK_STRENGTH,
                node_i,
                MAX_LINK_UTIL,
                save_matrix=False,
            )
            print(attacked_edges)
            continuous_series[i + 1] += attack_matrix

        outfile = (
            SCRIPT_HOME
            + "/data/traffic/"
            + network
            + "_crossfire_every_node_"
            + ATTACK_STRENGTH_STRING
            + ".txt"
        )
        print("Saving traffic matrix to: {}".format(outfile))
        np.savetxt(outfile, continuous_series.astype(int), fmt="%i")


def find_target_link(path_file):
    """Finds a target link from a path file. Returns the number of times a link appears in a path, and the flows that target
        each link
    Args:
        path_file (str): absolute path to a the path file
    Returns:
        tuple: (0. edge_use:dict (edge:pair -> count:int), 1. edge_flows:dict (edge:pair -> flows:list:pair), 2. most_used_edge:int)
    """
    with open(path_file, "r") as fob:
        edge_use = defaultdict(int)
        edge_flows = defaultdict(list)
        most_used_edge = 0
        a, b = None, None
        for line in fob:
            if line.startswith("h"):
                host_line = line.split(" -> ")
                a, b = [h.strip(" \n:") for h in host_line]
                if ZERO_INDEXED:  # zero indexed nodes
                    a = str(int(a.strip("h")) - 1)
                    b = str(int(b.strip("h")) - 1)
                else:  # one indexed nodes
                    a = a.strip("h")
                    b = b.strip("h")

            if line.startswith("["):  # line contains a path
                path, percent = line.strip().split("@")
                path_edges = parse_edges(path)
                for edge in path_edges:
                    edge_use[edge] += 1
                    edge_flows[edge].append((a, b))
                    if edge_use[edge] == max(edge_use.values()):
                        most_used_edge = edge

    return edge_use, edge_flows, most_used_edge
