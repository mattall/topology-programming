""" JSON Schema
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
"""
import json

from onset.utilities.flows import write_flows_to_json

if __name__ == "__main__":
    with open("/home/matt/network_stability_sim/notes.json", "r") as fob:
        config = json.load(fob)

    for network in config:
        for txt_p, json_p in [
            ("BaseTMTxt", "BaseTMJSON"),
            ("BreakingBaselineTMTxt", "BreakingBaselineTMJSON"),
            ("BreakingONSETTMTxt", "BreakingONSETTMJSON"),
        ]:
            base_traffic_matrix = config[network]["BaseTMTimeSeries"]
            attack_traffic_matrix = config[network][txt_p]
            json_paths = config[network]["BasePathsJSON"]
            out_file = config[network][json_p]
            congestion_factor = 1.0
            target_link = config[network]["targetLink"]
            write_flows_to_json(
                base_traffic_matrix,
                attack_traffic_matrix,
                json_paths,
                out_file,
                [target_link],
                congestion_factor,
            )
            adapted_paths = config[network]["AdaptedPathsJSON"]
            tag = out_file.split("/")[-1][
                :5
            ]  # get last bit of file name (minus .json)
            adaptive_flows = (
                "/home/matt/ripple/simulator/topologies/"
                + network
                + "/"
                + tag
                + "AdaptivePaths.json"
            )
            write_flows_to_json(
                base_traffic_matrix,
                attack_traffic_matrix,
                json_paths,
                adaptive_flows,
                [target_link],
                congestion_factor,
            )
            config[network][json_p + "AdaptivePaths"] = adaptive_flows
            with open(
                "/home/matt/network_stability_sim/notes_v2.json", "w"
            ) as fob:
                json.dump(config, fob, indent=4)

    if 0:
        base_traffic_matrix = (
            "/home/matt/network_stability_sim/data/traffic/sprint.txt"
        )
        attack_traffic_matrix = "/home/matt/network_stability_sim/data/traffic/sprint_link_2_9_strength_11"
        json_paths = "/home/matt/ripple/simulator/topologies/sprint/path.json"
        out_file = (
            "/home/matt/ripple/simulator/topologies/sprint/attack_100G.json"
        )
        congestion_factor = 1.0
        write_flows_to_json(
            base_traffic_matrix,
            attack_traffic_matrix,
            json_paths,
            out_file,
            [(1, 5)],
            congestion_factor,
        )
