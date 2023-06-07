from sys import argv

from onset.similarity import calc_accuracy_as_dataframe
from onset.utilities.graph_utils import read_json_graph
import seaborn as sns
def main(argv):
    try:
        G_path = argv[1]
    except:
        G_path = "data/graphs/json/regional/ground_truth_regional.json"
        H_path = "data/graphs/json/regional/reconstruction_regional.json"

    G = read_json_graph(G_path, stringify=True)
    H = read_json_graph(H_path, stringify=True)
    
    acc_df = calc_accuracy_as_dataframe(G, H)

if __name__ == "__main__":
    main(argv)