'''
    Generates flows for the given topology.
    by Abduarraheem Elfandi
    and Matt Nance-Hall
'''
from sys import argv, exit
from onset.utilities.flows import generate_flows
from onset.utilities.recon_utils import write_flows

def main(argv):
    # for testing purposes
    try:
        topology_file = argv[1]
        min_flow = float(argv[2])
        max_flow = float(argv[3])

    except:
        print("usage: python3 network.py topology_file.gml min_flow max_flow [output_file]")
        exit()

    try:        
        output_file = argv[4]

    except:
        output_file = topology_file + 'flow.txt'
        print(f'Output file not provided {output_file} will be generated.')

    flows = generate_flows(topology_file, min_flow, max_flow)
    write_flows(flows, output_file=output_file)

if __name__ == "__main__":
    main(argv)