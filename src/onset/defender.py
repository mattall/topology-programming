from collections import defaultdict
from logging import error
from posix import listdir
import re
from os import path

from networkx.algorithms.centrality.betweenness import edge_betweenness
from onset.utilities.sysUtils import cast_pair_to_int, reindex_down, reindex_up
from onset.utilities.logger import logger
from onset.utilities.post_process import calc_percent_diff, get_candidate_links, read_result_val
from onset.utilities import SCRIPT_HOME, ZERO_INDEXED, parse_edges, find_target_link


class Defender():
    def __init__(self, network: str, bandwidth: str, method: str, use_heuristic: str, tag="", attack_proportion="") -> None:
        self.network = network
        self.fallow_transponders = bandwidth
        self.method = method
        self.use_heuristic = use_heuristic
        self.attack_proportion = attack_proportion
        self.tag = tag if not attack_proportion else tag + "_" + attack_proportion
        self.candidate_links = get_candidate_links(network, bandwidth, method, attack_proportion)
        self.cached_sims_root = self.edge_congestion_file = path.join(SCRIPT_HOME, 'data', 'results', '{}_add_circuit_{}_{}_{}'.format(
            self.network, self.method, self.fallow_transponders, self.attack_proportion))
        if tag == "":
            self.path_file = path.join(
                self.cached_sims_root, '__0', "paths", "ecmp_0")
            self.edge_congestion_file = path.join(
                self.cached_sims_root, '__0', "EdgeCongestionVsIterations.dat")
        else:
            self.path_file = path.join(tag, "paths", "ecmp_0")
            self.edge_congestion_file = path.join(
                tag, "EdgeCongestionVsIterations.dat")

    def get_circuits_that_reduce_congestion_on_uv(self, u, v) -> dict:
        priority_links = {}
        # Circuit tag (a,z) -> percent reduction (float)
        links_that_reduce_congestion_on_uv = {}
        base_tag = "__0"
        metric_tag = "EdgeCongestionVsIterations.dat"
        congestion_naut_file = path.join(
            self.cached_sims_root, base_tag, metric_tag)
        edge_congestion_naut = self.get_edge_congestion(congestion_naut_file)
        uv_congestion_naut = edge_congestion_naut[(u, v)]
        circuit_regex = re.compile('^[0-9]+_[0-9]+$')
        circuit_tags = [tag for tag in listdir(
            self.cached_sims_root) if circuit_regex.match(tag)]
        for circuit_tag in circuit_tags:
            congestion_file = path.join(self.cached_sims_root,
                                        circuit_tag, metric_tag)
            edge_congestion_with_new_circuit = self.get_edge_congestion(
                congestion_file)
            uv_congestion_with_new_circuit = edge_congestion_with_new_circuit[(
                u, v)]
            if uv_congestion_with_new_circuit < uv_congestion_naut:
                node_a, node_b = circuit_tag.split("_")
                percent_diff = calc_percent_diff(
                    uv_congestion_naut, uv_congestion_with_new_circuit)
                if self.link_churn_on_path((node_a, node_b), (u, v)) > 0:
                    priority_links[node_a, node_b] = percent_diff
                links_that_reduce_congestion_on_uv[node_a,
                                                   node_b] = percent_diff

        if priority_links:
            return priority_links
        else:
            return links_that_reduce_congestion_on_uv

    def get_circuits_that_change_total_paths(self, positive: bool) -> dict:
        priority_links = {}
        # Circuit tag (a,z) -> reduction (int)
        circuits_that_change_total_paths = {}
        base_tag = "__0"
        metric_tag = "NumPathsVsIterations.dat"
        total_paths_naut_file = path.join(
            self.cached_sims_root, base_tag, metric_tag)
        total_paths_naut = read_result_val(total_paths_naut_file)
        circuit_regex = re.compile('^[0-9]+_[0-9]+$')
        circuit_tags = [tag for tag in listdir(
            self.cached_sims_root) if circuit_regex.match(tag)]
        for circuit_tag in circuit_tags:
            node_a, node_b = circuit_tag.split("_")
            total_paths_file = path.join(self.cached_sims_root,
                                         circuit_tag, metric_tag)
            total_paths_with_new_circuit = read_result_val(total_paths_file)
            path_diff = total_paths_with_new_circuit - total_paths_naut
            if positive:
                if path_diff > 0:
                    circuits_that_change_total_paths[node_a, node_b] = path_diff
            if not positive:
                if path_diff < 0:
                    circuits_that_change_total_paths[node_a, node_b] = path_diff

        return circuits_that_change_total_paths

    def get_edge_congestion(self, edge_congestion_file) -> dict:
        edge_congestion = {}
        core_link_regex = re.compile("^\s*\(s[0-9]+,s[0-9]+\)")
        with open(edge_congestion_file) as ecf:
            for line in ecf:
                if not core_link_regex.match(line):
                    continue
                edge, congestion = line.split(':')
                edge = edge.strip().strip('()').replace('s', '')
                edge = tuple(edge.split(','))
                congestion = float(congestion.strip())
                edge_congestion[edge] = congestion

        return edge_congestion

    def get_strategic_circuit(self):
        if self.method == 'heuristic':
            if self.use_heuristic == '1':  # Link that reduces the congestion on most congested edge
                edge_congestion = self.get_edge_congestion(
                    self.edge_congestion_file)
                try:
                    self.most_used_edge = max(
                        edge_congestion, key=edge_congestion.get)
                except ValueError:
                    return
                logger.debug("Searching for circuits to relieve congestion on link: {}".format(
                    self.most_used_edge))
                candidate_links = self.get_circuits_that_reduce_congestion_on_uv(
                    *self.most_used_edge)
                try:
                    most_reducing_link = min(
                        candidate_links, key=candidate_links.get)
                except(ValueError):
                    return
                # most_reducing_link = reindex_down(most_reducing_link)
                return most_reducing_link

            if self.use_heuristic == "2":  # adds most possible paths to target link - statically chosen
                self.edge_use, self.edge_flows, self.most_used_edge = find_target_link(
                    self.path_file)
                logger.debug("Searching for circuits to relieve congestion on link: {}".format(
                    self.most_used_edge))
                churn = []
                for candidate in self.candidate_links:
                    churn.append(
                        self.link_churn_on_path(
                            (candidate[1], candidate[2]), self.most_used_edge))

                # TODO: If churn is 0 for all links then an arbitrary choice is made. Should probably do something else - or nothing at all...
                try:
                    most_churnfull_id = churn.index(max(churn))
                except ValueError:
                    return
                link_to_add = self.candidate_links[most_churnfull_id][1], self.candidate_links[most_churnfull_id][2]
                # link_to_add = reindex_down(link_to_add)
                # link_to_add = cast_pair_to_int(*link_to_add)
                return link_to_add                
            
            if self.use_heuristic == "3":  # removes most possible paths from target link - statically chosen
                self.edge_use, self.edge_flows, self.most_used_edge = find_target_link(
                    self.path_file)
                logger.debug("Searching for circuits to relieve congestion on link: {}".format(
                    self.most_used_edge))
                churn = []
                for candidate in self.candidate_links:
                    churn.append(
                        self.link_churn_on_path(
                            (candidate[1], candidate[2]), self.most_used_edge))

                # TODO: If churn is 0 for all links then an arbitrary choice is made. Should probably do something else - or nothing at all...
                try:
                    most_churnfull_id = churn.index(min(churn))
                except ValueError:
                    return
                link_to_add = self.candidate_links[most_churnfull_id][1], self.candidate_links[most_churnfull_id][2]
                # link_to_add = reindex_down(link_to_add)
                # link_to_add = cast_pair_to_int(*link_to_add)
                return link_to_add                

            if self.use_heuristic == "4":  # Link that introduces that greatest number of new paths
                candidate_links = self.get_circuits_that_change_total_paths(
                    positive=True)
                try:
                    most_additive_link = max(
                        candidate_links, key=candidate_links.get)
                except ValueError:
                    return
                # most_additive_link = reindex_down(most_additive_link)
                return most_additive_link

            if self.use_heuristic == "5":  # Link that removes that greatest number of new paths
                candidate_links = self.get_circuits_that_change_total_paths(
                    positive=False)
                try:
                    most_reductive_link = min(
                        candidate_links, key=candidate_links.get)
                except ValueError:
                    return
                # most_reductive_link = reindex_down(most_reductive_link)
                return most_reductive_link

            
            if self.use_heuristic == "6": # Introduce the new paths while adding greatest number of paths to a targeted link.
                edge_congestion = self.get_edge_congestion(
                    self.edge_congestion_file)
                try:
                    self.most_used_edge = max(
                        edge_congestion, key=edge_congestion.get)
                except ValueError:
                    return

                logger.debug("Searching for circuits to relieve congestion on link: {}".format(
                    self.most_used_edge))
                churn = {}
                for candidate in self.candidate_links:
                    candidate_circuit = (candidate[1], candidate[2])
                    churn[candidate_circuit] = self.link_churn_on_path(candidate_circuit, self.most_used_edge)
                
                path_additive_links=self.get_circuits_that_change_total_paths(positive=True)
                while churn:
                    most_churnfull_circuit = max(churn, key=churn.get)
                    if most_churnfull_circuit in path_additive_links:
                        return most_churnfull_circuit
                    
                    del churn[most_churnfull_circuit]
                        
                return
            
            
            if self.use_heuristic == "7": # Introduce the new paths while removing greatest number of paths to a targeted link.
                edge_congestion = self.get_edge_congestion(
                    self.edge_congestion_file)
                try:
                    self.most_used_edge = max(
                        edge_congestion, key=edge_congestion.get)
                except ValueError:
                    return

                logger.debug("Searching for circuits to relieve congestion on link: {}".format(
                    self.most_used_edge))
                churn = {}
                for candidate in self.candidate_links:
                    candidate_circuit = (candidate[1], candidate[2])
                    churn[candidate_circuit] = self.link_churn_on_path(candidate_circuit, self.most_used_edge)
                
                path_additive_links=self.get_circuits_that_change_total_paths(positive=True)
                if path_additive_links: 
                    while churn:
                        most_churnfull_circuit = min(churn, key=churn.get)
                        if most_churnfull_circuit in path_additive_links:
                            return most_churnfull_circuit
                        
                        del churn[most_churnfull_circuit]
                else: # No links add more paths, add link to reduce congestion
                    congestion_reducing_link = min(self.candidate_links) 
                    return ( int(congestion_reducing_link[1]) , 
                             int(congestion_reducing_link[2]) )

        else:
            raise(Exception, "UNDEFINED CASE")

            # self.edge_use, self.edge_flows, self.most_used_edge = find_target_link(
            #     self.path_file)
            # logger.debug("Searching for circuits to relieve congestion on link: {}".format(
            #     self.most_used_edge))
            # churn = []
            # for candidate in self.candidate_links:
            #     churn.append(
            #         self.link_churn_on_path(
            #             (candidate[1], candidate[2]), self.most_used_edge))

            # # TODO: If churn is 0 for all links then an arbitrary choice is made. Should probably do something else - or nothing at all...
            # try:
            #     most_churnfull_id = churn.index(max(churn))
            # except ValueError:
            #     return
            # link_to_add = self.candidate_links[most_churnfull_id][1], self.candidate_links[most_churnfull_id][2]
            # # link_to_add = reindex_down(link_to_add)
            # # link_to_add = cast_pair_to_int(*link_to_add)
            # return link_to_add

    def link_churn_on_path(self, circuit: tuple, link: tuple):
        '''
        Returns the number of times a given link is added/removed from network paths due to adding the given circuit.
        '''
        logger.debug("circuit: ({}, {})".format(circuit[0], circuit[1]))
        path_diff_file = path.join(
            self.cached_sims_root, 'path_diff', "{}_{}.txt".format(circuit[0], circuit[1]))

        path_lost_regex = re.compile(
            '^<.*\(s{},s{}\).*'.format(link[0], link[1]))
        path_gained_regex = re.compile(
            '^>.*\(s{},s{}\).*'.format(link[0], link[1]))
        lost = 0
        gained = 0

        with open(path_diff_file) as fob:
            for line in fob:
                if path_gained_regex.match(line):
                    gained += 1
                if path_lost_regex.match(line):
                    lost += 1

        link_churn = gained - lost
        logger.debug("paths gained {}".format(gained))
        logger.debug("paths   lost {}".format(lost))
        logger.debug("churn       {}".format(link_churn))
        return link_churn


if __name__ == "__main__":
    defender = Defender('fishnet', '3')
    defender.get_strategic_circuit()
