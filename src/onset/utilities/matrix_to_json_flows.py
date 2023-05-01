''' JSON Schema
{
    "effective cong": {
        "(NODE_ID, NODE_ID)": float,
        ...
    },
    "nclass": int,
    "nlink": int.
    "target_link": {
        "link0": {
            "dst": NODE_STR,
            "stc": NODE_STR
        }
        ...
    },
    "traffic_class": {
        "tc0": {
            "dst": NODE_STR,
            "malicious": bool,
            "nhops": int,
            "rolling", int,
            "hops": [
                NODE_STR, 
                NODE_STR,
                ...
            ],
            "src": NODE_STR,
            "flow_rate": int (bits-per-second)
        },
        ...
    }
}
'''
from numpy import sqrt
from numpy import loadtxt
import json

    # Read data from traffic matrix
def read_tm_to_tc(tc:dict, tm:list, paths:dict, malicious:bool, rolling:int):
    """
    Args:
        tc (dict): traffic classes object, can be empty. 
        tm (list): traffic matrix assumed to be passed as a 1-D list
        paths (dict): paths object
        malicious (bool): traffic being read is malicious or not
        rolling (int): -1 if traffic is not malicious. Otherwise, which phase of attack is the traffic from. 
    """        
    tc_index = len(tc)
    dimension = int(sqrt(len(tm)))

    for entry in range(len(tm)):
        rate = int(tm[entry] // 1000)
        i = (entry // dimension) + 1
        j = (entry % dimension) + 1
        if i == j: continue            
        
        dst = "s" + str(j)
        src = "s" + str(i)
        flow_paths = {}
        for path in paths: 
            if paths[path]['src'] == src and paths[path]['dst'] == dst:
                flow_paths[path] = paths[path]
        
        npaths = len(flow_paths)
        for flow_path in flow_paths:    
            path_rate = rate // npaths
            if path_rate == 0:
                continue
            
            tc["tc{}".format(tc_index)] = {"dst": dst, 
                                           "malicious": malicious,
                                           "nhops":paths[flow_path]['nhop'],
                                           "rolling": rolling,
                                           "hops": paths[flow_path]['hops'],
                                           "src": src, 
                                           "flow_rate": int(rate // npaths)}
            tc_index += 1

def write_flows_to_json(base_traffic_matrix, attack_traffic_matrix, json_paths, out_file, targets, congestion_factor):
    # For each source, destination in the traffic matrix, 
    #   map source, destionation to a (set of) path(s).

    # Load traffic matrix and path data. 
    base_tm = loadtxt(base_traffic_matrix, dtype=int)
    attack_tm = loadtxt(attack_traffic_matrix, dtype=int)
    path_obj = json.load(open(json_paths, 'r'))
    paths = path_obj['paths']    
    # Prepare flow object meta-data
    flow_obj = {}
    target_links = [str(t) for t in targets]
    flow_obj["effective cong"] = {target_link:congestion_factor for target_link in target_links}
    flow_obj["target_link"] = {"link{}".format(i):{ "dst": "s{}".format(targets[i][1]), "src": "s{}".format(targets[i][0])} for i in range(len(targets))}
    flow_obj["nlink"] = len(targets)
    tc = {}
    read_tm_to_tc(tc, base_tm, paths, False, -1)
    read_tm_to_tc(tc, attack_tm, paths, True, 1)
    flow_obj["traffic_class"] = tc
    with open(out_file, 'w') as fob:
        json.dump(flow_obj, fob, indent=4)

if __name__ == "__main__":
    with open("/home/matt/network_stability_sim/notes.json", 'r') as fob:
        config = json.load(fob)

    for network in config: 
        for txt_p, json_p in [('BaseTMTxt', 'BaseTMJSON'), 
                            ("BreakingBaselineTMTxt","BreakingBaselineTMJSON"), 
                            ("BreakingONSETTMTxt", "BreakingONSETTMJSON")]:
            base_traffic_matrix = config[network]['BaseTMTimeSeries']
            attack_traffic_matrix = config[network][txt_p]
            json_paths = config[network]['BasePathsJSON']
            out_file = config[network][json_p]
            congestion_factor = 1.0
            target_link = config[network]['targetLink']
            write_flows_to_json(base_traffic_matrix, attack_traffic_matrix, json_paths, out_file, [target_link], congestion_factor)
            adapted_paths = config[network]['AdaptedPathsJSON']
            tag = out_file.split('/')[-1][:5] # get last bit of file name (minus .json)
            adaptive_flows = "/home/matt/ripple/simulator/topologies/" + network + "/" + tag + "AdaptivePaths.json"
            write_flows_to_json(base_traffic_matrix, attack_traffic_matrix, json_paths, adaptive_flows, [target_link], congestion_factor)
            config[network][json_p+"AdaptivePaths"] = adaptive_flows
            with open("/home/matt/network_stability_sim/notes_v2.json", 'w') as fob:
                json.dump(config, fob, indent=4)

    if 0: 
        base_traffic_matrix = "/home/matt/network_stability_sim/data/traffic/sprint.txt"
        attack_traffic_matrix = "/home/matt/network_stability_sim/data/traffic/sprint_link_2_9_strength_11"
        json_paths = "/home/matt/ripple/simulator/topologies/sprint/path.json"
        out_file = "/home/matt/ripple/simulator/topologies/sprint/attack_100G.json"
        congestion_factor = 1.0
        write_flows_to_json(base_traffic_matrix, attack_traffic_matrix, json_paths, out_file, [(1,5)], congestion_factor)


