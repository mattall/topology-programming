TOPOLOGY_GROUND_TRUTH="data/graphs/json/campus/campus_ground_truth.json"
N_FLOWS=1000
FLOWS_FILE="data/flows/campus_flows_min-${N_FLOWS}_max-${N_FLOWS}.csv"
python -c "from onset.utilities.flows import generate_flows; generate_flows('${TOPOLOGY_GROUND_TRUTH}',${N_FLOWS},${N_FLOWS},'${FLOWS_FILE}')"
python -m onset.utilities.flow_distribution ${TOPOLOGY_GROUND_TRUTH} ${FLOWS_FILE}
