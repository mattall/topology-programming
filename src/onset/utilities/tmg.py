# Creates a series of traffic matrices (TMs).
# Expected input: 1) Number of hosts in the network
#                 2) Type of TM to generate 
# 
# Output: A file whose lines are traffic matrices. 

from numpy import set_printoptions, savetxt, set_printoptions, transpose, reshape, mean
from numpy.random import pareto, exponential
import numpy as np

import sys
from sys import argv, exit
from os import makedirs

from tmgen.models import spike_tm, gravity_tm, random_gravity_tm, modulated_gravity_tm

def percent_diff(a, b):
    if a == b: 
        return 0
        
    return (abs(a - b) / ((a + b) / 2)) * 100

def make_human_readable(number:int) -> str:
    # inverse of sanitize magnitude
    # tera = number // 10 ** 12
    # if tera > 0:
    #     return str(tera) + "Tbps"
    if number == 0:
        return '0Gbps'

    giga = number // 10 ** 9
    if giga > 0:
        return str(giga) + "Gbps"
    mega = number // 10 ** 6
    if mega > 0:
        return str(mega) + "Mbps"
    kilo = number // 10 ** 3
    if kilo > 0:
        return str(kilo) + "Kbps"
    else:
        return str(number) + "bps"
    

def sanitize_magnitude(mag_arg: str) -> int:
    """Converts input magnitude arg into an integer
        Unit identifier is the 4th from list character in the string, mag_arg[-4].
        e.g., 1231904Gbps G is 4th from last.
        this returns 1231904 * 10**9.
    Args:
        mag_arg (str): number joined with either T, G, M, or K.

    Returns:
        int: Value corresponding to the input.
    """
    if mag_arg =='0bps':
        return 0

    mag = mag_arg[-4].strip()
    coefficient = int(mag_arg[0:-4])
    print("Coefficient : {}".format(coefficient))
    print("Magnitude   : {}".format(mag))
    exponent = 0
    if mag == 'T':
        exponent = 12
    elif mag == 'G':
        exponent = 9
    elif mag == 'M':
        exponent == 6
    elif mag == 'k':
        exponent == 3
    else:
        raise("ERROR: ill formed magnitude argument. Expected -m <n><T|G|M|k>bps, e.g., 33Gbps")
    result = coefficient * 10 ** exponent
    print("Result: {}".format(result))
    return result

set_printoptions(suppress=True)

def save_tm(tm, filename):
    '''
    @param [[x,],]  tm      :   Traffic matrix as list of lists
    @param string   filename:   Name for TM file as a path

    @returns: None
    '''
    with open(filename, 'w') as fob:
        for line in tm:
            for i in line:
                # fob.write("{:.4f} ".format(i))
                fob.write("{} ".format(i))
            fob.write("\n")
       
def rand_gravity_matrix(hosts:int, epochs:int, expected_mean:float=0.1, name:str = "rand_gravity_matrix"):  
    """ Creates a random gravity model matrix and saves it with the desired name.

    Args:
        hosts (int): number of hosts
        epochs (int, optional): How many TMs to generate. Defaults to 1.
        expected_mean (float): average demand between any two hosts (Gbps).
        name (str, optional): [description]. Defaults to "".
    """    
    
    if name == "": name = str(size)
    tm = []
    for _ in range(epochs):
        tm_rows = random_gravity_tm(hosts, expected_mean).matrix
        tm_i = []
        for row in tm_rows:
            for i in row:
                tm_i.append(int(i))    
        
        tm.append(tm_i)
        assert percent_diff(expected_mean, sum(tm_i)) < 0.01, "Expected mean is off by more than 0.01\%"
    save_tm(tm, name)
    print("saved traffic matrix to: {}".format(name))
    return np.array(tm)

def modulated_gravity_matrix(hosts:int, epochs:int, expected_mean:float=0.1, name:str = "modulated_gravity_matrix", save=True):  
    """ Creates a modulated gravity model matrix and saves it with the desired name.

    Args:
        hosts (int): number of hosts
        epochs (int, optional): How many TMs to generate. Defaults to 1.
        expected_mean (float): average demand between any two hosts (Gbps).
        name (str, optional): [description]. Defaults to "".
    """    
    # print(type(hosts), hosts)
    # print(type(epochs), epochs)
    # print(type(expected_mean), expected_mean)

    tm = modulated_gravity_tm(hosts, epochs, expected_mean * hosts)
    tm = tm.matrix.reshape(tm.num_epochs(), tm.num_nodes() ** 2 )
    if save:
        savetxt(name, tm, delimiter=' ')
    return tm 

def pareto_matrix(size, expected_mean, length = 1, outdir = "./", name = ""):
    '''
    @param int      size    :   number of nodes in the network
    @param int      length  :   number of TMs in series to generate
    @param string   out_dir :   The folder in which to save the resulting TM.
    @param string   name    :   name for TM file

    @returns: None
    '''
    if name == "": name = str(size)
    outFile = outdir + name + "_pareto-matrix.txt"

    tm = []
    for _ in range(length):
        tm_i = ((pareto(1,size ** 2) + 1 ) * 1)
        sample_mean = mean(tm_i)
        tm_i *= expected_mean / sample_mean * tm_i
        tm.append(tm_i)

    save_tm(tm, outdir, outFile, name, size)
    
def constant_matrix(size, expected_mean, length = 1, outdir = "./", name = ""):
    '''
    @param int      size    :   number of nodes in the network
    @param int      length  :   number of TMs in series to generate
    @param string   out_dir :   The folder in which to save the resulting TM.

    @returns: None
    '''
    if name == "": name = str(size)
    outFile = outdir + str(name) + "_constant-matrix.txt"

    tm = []
    for _ in range(length):
        tm_i = [expected_mean] * size ** 2
        tm.append(tm_i)

    save_tm(tm, outdir, outFile, name, size)

def exponential_matrix(size, expected_mean, length = 1, outdir = "./", name = ""):
    if name == "": name = str(size)
    outFile = outdir + str(name) + "_exponential-matrix.txt"

    print("MEAN: {} LENGTH: {} SIZE: {}".format(expected_mean, length, size))
    m = np.random.exponential(expected_mean, (length, size ** 2))
    save_tm(m, outdir, outFile, name, size)


def main(argv):
    def print_usage():
        print("usage: python tmg.py <NAME> <NUM_HOSTS> <OUTPUT_DIR>")

    try:
        topology = argv[1]
    except:
        print("Got: {}".format(argv))
        print("Missing NAME")
        print_usage()
        exit()

    try:
        hosts = int(argv[2])
    except:
        print("Got: {}".format(argv))
        print("Missing NUM_HOSTS")
        print_usage()
        exit()

    try:
        out_dir = argv[3]
    except:
        print("Got: {}".format(argv))
        print("Missing OUTPUT_DIR")
        print_usage()
        exit()

    # t = (topology, hosts)
    # tm = modulated_gravity_tm(t[1],200,1000)
    # m = tm.matrix.reshape(( tm.num_epochs(), tm.num_nodes() ** 2 ))
    # savetxt(out_dir + "./{}_gravity-matrix.txt".format(t[0]), m, fmt="%4.4f", delimiter=' ')

    # tm = exp_tm(t[1],1000,200)
    # m = tm.matrix.reshape(( tm.num_epochs(), tm.num_nodes() ** 2 ))
    # savetxt(out_dir + "./{}_exponential-matrix.txt".format(t[0]), m, fmt="%4.4f", delimiter=' ')

    pareto_matrix(hosts, 200, out_dir, topology)
    constant_matrix(hosts, 200, out_dir, topology)
    # modulated_gravity_matrix(hosts, 200, out_dir, topology)
    rand_gravity_matrix(hosts, 200, out_dir, topology)

if __name__=="__main__":
    main(argv)