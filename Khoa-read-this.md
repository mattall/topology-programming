# Skinwalker - Optical Topology Programming Defense Against Network Reconnaissance


## 1: Show vulnerability and attack using NOC dataset.

### 1.1: Background information. 

There is a script for parsing `show interface` command output in `topology-programming/data/NOC/interface_log_parser.py`

Another script, aggregate_logs_to_graph.py, in the same folder, uses a rudimentary clustering technique, Affinity Propagation, to cluster log data from different routers and suggest which pairs of interfaces could be a direct link between the routers. 

One should be able to use these scripts, and improve on them, to demonstrate topology recon for other networks. 

As an example, topology-programming/data/NOC/sox.edu.iu.grnoc.routerproxy, has the a folder for each router in the SOX network (https://routerproxy.grnoc.iu.edu/sox/). 
Note: The routers have been removed from the link above, and are not accessible as of today (9/22/23). The data in the folder was collected on 6/15/23.

The in each folder, there is a showint.log file, that has the output of the `show interfaces` command. 

the corresponding `.json` file in the directory has the result of parsing the data with  `interface_log_parser.py`.

The file, `sox.edu.iu.grnoc.routerproxy/Interface_clusters.txt`, shows the result of running `aggregate_logs_to_graph.py`. 

- You CANNOT take the output here as any definitive proof that two interfaces on two routers are in fact connected.

- The ONLY reason these clusters exist is because of similarity between the strings that describe the interfaces on two routers. 

- You need to manually examine these outputs to make your best guess 

### 1.2: TO-DO.

Choose a large network, e.g., AREON, https://routerproxy.areon.net/areon/,
and uncover as much topology information on the network as possible by using router-command from the publicly open NOC.

Follow the procedure described above to accomplish this task. 

Feel free to make any necessary changes to the procedure to complete the task. 

The result should be a graph describing the network layer connections and their bandwidth between the routers. 

### 1.3: Bonus.

Can you use a large-language-model (LLM), e.g., ChatGPT or custom-tuned variant, to do all of this work reliably? 

You should first do the ground work yourself to find how accurate of a map you can make of the network. Then feed the data (e.g., showinterfaces.log) into an LLM and judge how accurately it can reach the same conclusions as you, offer deeper verifiable insights, or whether it just completely "hallucinates", i.e., gives verifiable false/wildly inaccurate information. 

## 2: Evaluate Skinwalker's performance against the attack above.

### 2.1: Background information.

The code in `topology-programming/src/onset` is an optical topology programming simulator. 

A description of how it works is detailed in our TDSC submission attached (Section 5).

The simulator itself is in `topology-programming/src/onset/simulator.py` and you can import it to another python program with the following, `from onset.simulator import Simulation`.  You can see an example of how to run the simulator with various starting parameters in `scripts/experiment.py`

The program `python src/onset/net_sim.py` is the preferred command-line access point for the simulator---You usually just use this if you want to quickly test a one-off experiment, otherwise you will import Simulator into your program directly and pass in all of the required parameters for your experiment there.

- You can run this program on the command line, by calling  and it will give you USAGE and HELP information if you call it incorrectly or run `python src/onset/net_sim.py --help`. 

- At minimum, `net_sim.py` needs the name of a network, NAME, the number of hosts, and a string description of the experiment. 

- The network needs to have a file, either `NAME.gml` or `NAME.dot` in `topology-programming/data/graphs/gml` or `topology-programming/data/graphs/dot`.


### 2.2. An aside regarding traffic matrices, and classes thereof.

A traffic matrix is a square, `NxN` matrix,  `M` where each entriy, `(i,j)` in `M` describes the demand (in bits) from a host, `i` to host `j`. 

*classes of traffic* example, the (arguably) simplest traffic matrix has uniform demand between all pairs of hosts, e.g., `M[i,j] = c` for all `i` and all `j`.

We would like to have *real-world* traffic matrices whenever possible, but that is not always the case.

You can look at `https://github.com/mattall/TMgen` for examples and descriptions of various classes of traffic matrices.

### 2.3: TO-DO.

1. Create your own fork of `https://github.com/mattall/topology-programming`

2. Follow the instruction to install it in your workspace (make sure you use pip version 22.3.1 with you install with `pip install .`).

3. You should run the the simulator, from the command line or by importing it to your own working script, to get familiar with the execution flow of the program. 

4. Run the program with different graphs, found in `data/graphs/**` to determine which classes of traffic matrices can effectively leverage `skinwalker` to introduce differences to the topology while maintaining traffic performance of 100% throughput for all traffic.