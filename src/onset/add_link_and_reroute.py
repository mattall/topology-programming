'''
    Identify 'n' targets of link flood attack
    For each target, identify the 'm' largest contributing flows
        Find the most common sub-path of two or more links shared among the flows.
            Method, start at target link and iterate outwards, BFS.
        add a new link bypassing this sub-path 
        Add re-route path rules to split traffic onto the new path. 
            Maximize load for the new link

Inputs. 
1. Attack traffic matrix. 
2. Congestion file.
'''
from utilities.write_gml import write_gml
from collections import defaultdict
from os import path
import pandas
import networkx as nx
import json


def is_subpath(a, b, input_path, distance=1):
    # Returns True if 'a' and 'b' are a subpath of 'path' separated by a fixed 'distance' whose default is 1.
    if a in input_path and b in input_path:
        first = input_path.index(a)
        second = input_path.index(b)
        if first + distance == second:
            return True
    return False

def main():

    experiment = "linear_10_smc10" # source link, multihop, congestion constrained, 10 link
    if experiment == "ANS_smc10":
        TOPO                    = '/home/matt/ripple/simulator/topologies/ANS/ANS_1-20.gml'
        UPDATED_TOPO            = '/home/matt/network_stability_sim/data/graphs/ANS_Updated_source_link_only.gml'
        PATH_DB                 = '/home/matt/network_stability_sim/data/paths/ANS.json'
        ATTACK                  = '/home/matt/network_stability_sim/data/traffic/ANS_targets_10_strength_10.json'
        CONGESTION              = '/home/matt/ripple/simulator/logs/ANS_1Tbps_ripple/link_util.csv'
        REROUTED_PATHS          = '/home/matt/network_stability_sim/data/paths/ANS_Rerouted_source_link_only.json'
        NEW_LINK_MAX_HOPS       = 3
        N_TARGET_LINKS          = 10
        MAX_LINK_UTIL           = 0.95
        SOURCE_LINK_ADDITION    = False # new links start from the target link's source node
        REPEAT = False
    
    if experiment == "sprint_smc10":
        TOPO                    = '/home/matt/ripple/simulator/topologies/sprint/sprint_1-20.gml'
        UPDATED_TOPO            = '/home/matt/network_stability_sim/data/graphs/sprint_Updated_source_link_only.gml'
        PATH_DB                 = '/home/matt/network_stability_sim/data/paths/sprint.json'
        ATTACK                  = '/home/matt/network_stability_sim/data/traffic/sprint_targets_10_strength_10.json'
        CONGESTION              = '/home/matt/ripple/simulator/logs/sprint_1Tbps_ripple/link_util.csv'
        REROUTED_PATHS          = '/home/matt/network_stability_sim/data/paths/sprint_Rerouted_source_link_only.json'
        NEW_LINK_MAX_HOPS       = 3
        N_TARGET_LINKS          = 10
        MAX_LINK_UTIL           = 0.95
        SOURCE_LINK_ADDITION    = False # new links start from the target link's source node
        REPEAT = False

    if experiment == "grid_3_smc10":
        TOPO                    = '/home/matt/network_stability_sim/data/graphs/gml/grid_3.gml'
        UPDATED_TOPO            = '/home/matt/network_stability_sim/data/graphs/grid_3_Updated_source_link_only.gml'
        PATH_DB                 = '/home/matt/network_stability_sim/data/paths/grid_3.json'
        ATTACK                  = '/home/matt/network_stability_sim/data/traffic/grid_3_targets_10_strength_10.json'
        CONGESTION              = '/home/matt/ripple/simulator/logs/grid_3_ripple/link_util.csv'
        REROUTED_PATHS          = '/home/matt/network_stability_sim/data/paths/grid_3_Rerouted_source_link_only.json'
        NEW_LINK_MAX_HOPS       = 3
        N_TARGET_LINKS          = 10
        MAX_LINK_UTIL           = 0.95
        SOURCE_LINK_ADDITION    = False # new links start from the target link's source node
        REPEAT = False

    if experiment == "linear_10_smc10":
        TOPO                    = '/home/matt/network_stability_sim/data/graphs/gml/linear_10.gml'
        UPDATED_TOPO            = '/home/matt/network_stability_sim/data/graphs/linear_10_Updated_source_link_only.gml'
        PATH_DB                 = '/home/matt/network_stability_sim/data/paths/linear_10.json'
        ATTACK                  = '/home/matt/network_stability_sim/data/traffic/linear_10_targets_10_strength_10.json'
        CONGESTION              = '/home/matt/ripple/simulator/logs/linear_10_ripple/link_util.csv'
        REROUTED_PATHS          = '/home/matt/network_stability_sim/data/paths/linear_10_Rerouted_source_link_only.json'
        NEW_LINK_MAX_HOPS       = 3
        N_TARGET_LINKS          = 10
        MAX_LINK_UTIL           = 0.95
        SOURCE_LINK_ADDITION    = False # new links start from the target link's source node
        REPEAT = False

    if experiment == "whisker_3_2_smc10":
        TOPO                    = '/home/matt/network_stability_sim/data/graphs/gml/whisker_3_2.gml'
        UPDATED_TOPO            = '/home/matt/network_stability_sim/data/graphs/whisker_3_2_Updated_source_link_only.gml'
        PATH_DB                 = '/home/matt/network_stability_sim/data/paths/whisker_3_2.json'
        ATTACK                  = '/home/matt/network_stability_sim/data/traffic/whisker_3_2_targets_8_strength_10.json'
        CONGESTION              = '/home/matt/ripple/simulator/logs/whisker_3_2_ripple/link_util.csv'
        REROUTED_PATHS          = '/home/matt/network_stability_sim/data/paths/whisker_3_2_Rerouted_source_link_only.json'
        NEW_LINK_MAX_HOPS       = 3
        N_TARGET_LINKS          = 10
        MAX_LINK_UTIL           = 0.95
        SOURCE_LINK_ADDITION    = False # new links start from the target link's source node
        REPEAT = False


    elif experiment == "ANS_amc10": # any link, multihop, congestion constrained, 10 link
        TOPO                    = '/home/matt/ripple/simulator/topologies/ANS/ANS_1-20.gml'
        UPDATED_TOPO            = '/home/matt/network_stability_sim/data/graphs/ANS_Updated_any_link.gml'
        PATH_DB                 = '/home/matt/network_stability_sim/data/paths/ANS.json'
        ATTACK                  = '/home/matt/network_stability_sim/data/traffic/ANS_targets_10_strength_10.json'
        CONGESTION              = '/home/matt/ripple/simulator/logs/ANS_1Tbps_ripple/link_util.csv'
        REROUTED_PATHS          = '/home/matt/network_stability_sim/data/paths/ANS_Rerouted_any_link.json'
        NEW_LINK_MAX_HOPS       = 3
        N_TARGET_LINKS          = 10
        MAX_LINK_UTIL           = 0.95
        SOURCE_LINK_ADDITION    = False # new links start from the target link's source node
        REPEAT = False

    elif experiment == "ANS_smc10_repeat": # any link, multihop, congestion constrained, 10 link
        # TODO
        TOPO                    = '/home/matt/network_stability_sim/data/graphs/ANS_Updated_any_link.gml'
        UPDATED_TOPO            = ''
        PATH_DB                 = '/home/matt/network_stability_sim/data/paths/ANS_Rerouted_any_link.json'
        ATTACK                  = '/home/matt/network_stability_sim/data/traffic/ANS_targets_10_strength_10.json'
        CONGESTION              = '/home/matt/ripple/simulator/logs/ANS_1Tbps_ripple/link_util.csv'
        REROUTED_PATHS          = ''
        NEW_LINK_MAX_HOPS       = 3
        N_TARGET_LINKS          = 10
        MAX_LINK_UTIL           = 0.95
        SOURCE_LINK_ADDITION    = False # new links start from the target link's source node
        REPEAT = False

    
    elif experiment == "ANS_amc10": # any link, multihop, congestion constrained, 10 link
        TOPO                    = '/home/matt/ripple/simulator/topologies/ANS/ANS_1-20.gml'
        UPDATED_TOPO            = '/home/matt/network_stability_sim/data/graphs/ANS_Updated_any_link.gml'
        PATH_DB                 = '/home/matt/network_stability_sim/data/paths/ANS.json'
        ATTACK                  = '/home/matt/network_stability_sim/data/traffic/ANS_targets_10_strength_10.json'
        CONGESTION              = '/home/matt/ripple/simulator/logs/ANS_1Tbps_ripple/link_util.csv'
        REROUTED_PATHS          = '/home/matt/network_stability_sim/data/paths/ANS_Rerouted_any_link.json'
        NEW_LINK_MAX_HOPS       = 3
        N_TARGET_LINKS          = 10
        MAX_LINK_UTIL           = 0.95
        SOURCE_LINK_ADDITION    = False # new links start from the target link's source node
        REPEAT = False

    source_files = [TOPO, PATH_DB, ATTACK, CONGESTION] 
    for sf in source_files:
        assert path.exists(sf)

    if REPEAT:
            TOPO = UPDATED_TOPO
            PATH_DB = REROUTED_PATHS
            CONGESTION = '/home/matt/ripple/simulator/logs/ANS_1Tbps_ripple_reroute_onset/link_util.csv'


    # READ CONGESTION FILE
    congestion_df = pandas.read_csv(CONGESTION)
    time_t = 6.0
    congestion_at_time_t = congestion_df.loc[congestion_df['time']==6.0].drop('time', axis=1).stack().sort_values(ascending=False)

    # IDENTIFY TARGET LINKS
    target_links = []
    for multi_index in congestion_at_time_t.index[0:N_TARGET_LINKS]:
        _, link = multi_index
        a, b = link.strip("()").replace(" ", '').replace("'", '').split(',')
        target_links.append((a,b))

    # Load topology
    G = nx.read_gml(TOPO)

    # Load traffic traces
    with open(ATTACK, 'r') as fob:
        attack_db = json.load(fob)
        
    traffic = attack_db['traffic_class']

    new_links = set()
    for i, target_link in enumerate(target_links):
        if target_link == ('s9', 's14'):
            print("Problem link.")
        print('target link: {}'.format(target_link))
        subpaths = defaultdict(int)
        for tc in traffic:
            traffic_path = traffic[tc]['hops']
            if target_link[0] in traffic_path and target_link[1] in traffic_path:
                first = traffic_path.index(target_link[0])
                second = traffic_path.index(target_link[1])
                if  1 == second - first: # Target link is on this path.
                    if NEW_LINK_MAX_HOPS <= 3:
                        prefixed_subpath = tuple(traffic_path[first-1:second+1])
                        postfixed_subpath = tuple(traffic_path[first:second+2])
                        if len(prefixed_subpath) == 3 and not SOURCE_LINK_ADDITION:
                            subpaths[prefixed_subpath] += traffic[tc]['flow_rate']
                        if len(postfixed_subpath) == 3:
                            subpaths[postfixed_subpath] += traffic[tc]['flow_rate']
                    
                    if NEW_LINK_MAX_HOPS == 3:
                        prefixed_subpath = tuple(traffic_path[first-2:second+1])
                        postfixed_subpath = tuple(traffic_path[first:second+3])
                        if len(prefixed_subpath) == 4 and not SOURCE_LINK_ADDITION:
                            subpaths[prefixed_subpath] += traffic[tc]['flow_rate']
                        if len(postfixed_subpath) == 4:
                            subpaths[postfixed_subpath] += traffic[tc]['flow_rate']

        attempt = 0
        while subpaths:
            attempt += 1
            most_heavily_shared_subpath = max(subpaths, key=subpaths.get)
            subpaths.pop(most_heavily_shared_subpath)
            candidate_link = most_heavily_shared_subpath[0],  most_heavily_shared_subpath[-1]
            link_key = tuple(sorted(candidate_link))
            if link_key not in new_links:
                new_links.add(link_key)
                G.add_edge(link_key[0], link_key[1], capacity=100)
                print("added link, {}, after {} attempt(s)\n".format(link_key, attempt)) 
                break
        
        
    LINK_CAP = 100000000
    # add bidirectional capacity for new links. 
    new_link_cap = {(a, b): LINK_CAP for (a, b) in new_links}
    new_link_cap.update({(b, a): LINK_CAP for (a, b) in new_links})

    # find the most common subpath, from traffic[:]['hops'] that includes each link.
    rerouted_tcs = {}


    for tc in traffic:
        if tc == 'tc230':
            print(tc)
        # if the traffic class contains an attacked link,
        # find one or two new links that can benefit it. 
        rerouted_tcs[tc] = {}
        rerouted_tcs[tc]['dst'] = traffic[tc]['dst']
        rerouted_tcs[tc]['src'] = traffic[tc]['src']
        rerouted_tcs[tc]['volume'] = traffic[tc]['flow_rate']
        rerouted_tcs[tc]['paths'] = {}
        rerouted_tcs[tc]['paths']['paths0'] = {}
        rerouted_tcs[tc]['paths']['paths0']["hops"] = traffic[tc]['hops'][:]
        rerouted_tcs[tc]['paths']['paths0']["hops"] 
        hops = rerouted_tcs[tc]['paths']['paths0']["hops"]
        
        del_cnt = 0 # just for internal checking, curious how many paths have multiple bypass links.
        # for each flow (tc), check every new links against tc's paths. 
        # if the path can use any of the new links, add them to the path. 
        tc_bypass_links = []
        for a, b in new_links:
            # Prioritize bypass links for longer hop distances first. 
            if NEW_LINK_MAX_HOPS == 3 and is_subpath(a, b, hops, distance=3):
                tc_bypass_links.append((a, b))
                dropped_node = traffic[tc]['hops'].index(a) + 1
                del hops[dropped_node]
                dropped_node = traffic[tc]['hops'].index(a) + 1
                del hops[dropped_node]
                del_cnt += 2

            # drop bypass nodes from path. 
            if is_subpath(a, b, hops, distance=2):
                tc_bypass_links.append((a, b))
                dropped_node = traffic[tc]['hops'].index(a) + 1
                del hops[dropped_node]
                del_cnt += 1
                
        

        if len(tc_bypass_links) >= 1: 
            # if new link has bandwidth, 
            
            for bypass_link in tc_bypass_links:
                bypass_link_cap = new_link_cap[bypass_link] * MAX_LINK_UTIL
                if bypass_link_cap >= traffic[tc]['flow_rate']:
                    #   Add flow to path with new link. 
                    rerouted_tcs[tc]['paths']['paths0']['nhop'] = len(hops) - 1
                    rerouted_tcs[tc]['paths']['paths0']["flowFraction"] = 1.0
                    #   Decrease count for flow bandwidth
                    new_link_cap[bypass_link] -= traffic[tc]['flow_rate']
                
                # if link has insufficient bandwidth, 
                elif bypass_link_cap > 0 and bypass_link_cap < traffic[tc]['flow_rate']: 
                    # find what fraction of demand can be allocated to the new link. 
                    fraction_available = bypass_link_cap  / traffic[tc]['flow_rate'] 
                    fraction_leftover = 1 - fraction_available
                    demand_allocated = fraction_available * traffic[tc]['flow_rate']

                    # update rerouted tcs and new_link_cap            
                    rerouted_tcs[tc]['paths']['paths0']["flowFraction"] = fraction_available 
                    new_link_cap[bypass_link] -= demand_allocated

                    # Use the original path for the remaining portion of demand. 
                    rerouted_tcs[tc]['paths']['path1'] = {}
                    rerouted_tcs[tc]['paths']['path1']["flowFraction"] = fraction_leftover
                    rerouted_tcs[tc]['paths']['path1']['nhop'] = len(traffic[tc]['hops']) - 1
                    rerouted_tcs[tc]['paths']['path1']['hops'] = traffic[tc]['hops'][:]
                
                else: # can't use bypass link
                    rerouted_tcs[tc]['paths']['paths0']['nhop'] = len(hops) - 1
                    rerouted_tcs[tc]['paths']['paths0']["flowFraction"] = 1.0

        else: # No bypass link available
            rerouted_tcs[tc]['paths']['paths0']['nhop'] = len(hops) - 1
            rerouted_tcs[tc]['paths']['paths0']["flowFraction"] = 1.0
                    

        rerouted_tcs[tc]['npath'] = len(rerouted_tcs[tc]['paths'])            

    write_gml(G, UPDATED_TOPO)
    print("wrote to new topology to: {}".format(UPDATED_TOPO))

    new_link_dict = {}
    for i, nl in enumerate(sorted(new_links)):
        src, dst = nl
        new_link_dict["link" + str(i)] = {}
        new_link_dict["link" + str(i)]["src"] = src
        new_link_dict["link" + str(i)]["dst"] = dst

    with open(REROUTED_PATHS, 'w') as fob:
        json.dump({ "new_link" :  new_link_dict,
                    "tcs": rerouted_tcs, 
                    "ntc" : len(rerouted_tcs)}, fob, indent=4)
    print("wrote to new paths to: {}".format(REROUTED_PATHS))


if __name__ == "__main__":
    main()