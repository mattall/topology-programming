# experiment_params.py 


te_methods = [
    # "semimcfraekeft",
    "mcf",
    # "ncflow"
]

tp_methods = [
    "TE", 
    "TBE", 
    "greylambda",
    "Doppler",
    ]

networks = [
    "b4",
    "Zayo",
    "azure",
    "Verizon",
    "Comcast",
    "Campus",
    "three_node",
    "four-node",
    "areon"
]

t_classes = [
    "background", 
    # "background-plus-flashcrowd",
    ]

TE_name = {
    "-mcf": "MCF",
    "-semimcfraekeft": "SMORE",
    "-ncflow": "NCFlow",
}

n_ftx = {
    "TE": 0, 
    "TBE": 0, 
    "greylambda": 2,
    "Doppler": 2,
    "OSNET": 10,
    }

hosts = {
    "Campus": 14,
    "three_node": 3,
    "Comcast": 149,
    "Tinet": 53,
    "Verizon": 116,
    "azure": 113,
    "Zayo": 96,
    "b4": 54,
    "four-node": 4,
    "areon": 10, 
    }

mcf_loss_factor = {
    "Comcast": {
        "background": 1.210373,
        "background-plus-flashcrowd": 1.633346,
        },
    "Tinet": {
        "background": 297.500000,
        },
    "Campus": {
        "background": 1.803336,
        # "background-plus-flashcrowd": 1.633346,
        },
    "three_node": {
        "background": 1.0,
        # "background-plus-flashcrowd": 1.633346,
        },
    "four-node": {
        "background": 1.0,
        # "background-plus-flashcrowd": 1.633346,
        },        
    "areon": {
        "background": 1.0,
        # "background-plus-flashcrowd": 1.633346,
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

repeat = {
    "TBE": False, 
    "greylambda": True, 
    "TE": False,
    "Doppler": False
    }
