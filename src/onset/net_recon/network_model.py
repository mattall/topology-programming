'''
    Given network topology as an input generates output to describe a physical and logic network.
    by Abduarraheem Elfandi
    and Matthew Nance-Hall
'''

import random

from onset.network_model import Network
from onset.constants import IPV4, IPV4LENGTH
from sys import argv, exit
import pprint

pp = pprint.PrettyPrinter(indent=4)


def main(argv):
    # for testing purposes
    try:
        topology_file = argv[1]
        # if not topology_file.endswith('.gml'):
        #   print('Input file is not gml format.')
        #   exit()
    except:
        print("usage: python3 network.py topology_file [output_file]")
        exit()
    try:
        output_file = argv[2]
    except:
        output_file = topology_file + '_netout' + '.json'
        print(f'Output file not provided {output_file} will be generated.')
    
    network = Network(topology_file, output_file)
    
    network.export_network_json()
    network.export_network_plot()

if __name__ == "__main__":
    main(argv)