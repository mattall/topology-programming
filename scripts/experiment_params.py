# experiment_params.py 


te_methods = [
    "semimcfraekeft",
    "mcf",
    # "ncflow"
]

tp_methods = ["TE", "TBE", "greylambda"]

networks = [
    # "b4",
    # "Zayo",
    # "azure",
    # "Verizon",
    "Comcast"
]

t_classes = ["background", "background-plus-flashcrowd"]

TE_name = {
    "-mcf": "MCF",
    "-semimcfraekeft": "SMORE",
    "-ncflow": "NCFlow",
}

n_ftx = {"TE": 0, "TBE": 0, "greylambda": 2}

hosts = {
    "Comcast": 149,
    # "Verizon": 116,
    # "azure": 113,
    # "Zayo": 96,
    # "b4": 54,
}

mcf_loss_factor = {
    "Comcast": {
        "background": 1.210373,
        "background-plus-flashcrowd": 1.633346,
    },
}

demand_scale = [
    "0.1",
    "0.2",
    "0.3",
    "0.4",
    "0.5",
    "0.6",
    "0.7",
    "0.8",
    "0.9",
    "1.0",
    "1.1",
    "1.2",
    "1.3",
    "1.4",
    "1.5",
]

repeat = {"TBE": False, "greylambda": True, "TE": False}
