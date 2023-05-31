from sys import argv, exit
from networkx.classes.graph import Graph
from onset.utilities.graph_utils import Gml_to_dot
from .logger import logger

def main(argv):
    try:
        gml_file = argv[1]
    except:
        print("usage: python gml_to_dot.py gml_file [output_file]")
        exit()
        
    try:
        outfile = argv[2]
    except:
        outfile = gml_file.split('.')[0]

    Gml_to_dot(gml_file, outfile)

if __name__ == "__main__":
    main(argv)

