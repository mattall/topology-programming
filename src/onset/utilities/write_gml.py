def write_gml(G, name):
    with open(name, 'w') as fob:
        fob.write('graph [\n')
        for node in sorted(G.nodes()):
            id = node.strip('s')
            fob.write('\tnode [\n')
            fob.write('\t\tid {}\n'.format(id))
            fob.write('\t\tlabel "{}"\n'.format(node))
            for key, value in G.nodes[node].items():
                fob.write('\t\t{} {}\n'.format(key, value))
            fob.write('\t]\n')
        for s, t in G.edges():
            src = s.strip('s')
            dst = t.strip('s')
            fob.write('\tedge [\n')
            fob.write('\t\tsource {}\n'.format(src))
            fob.write('\t\ttarget {}\n'.format(dst))
            for key, value in G.edges[s,t].items():
                fob.write('\t\t{} {}\n'.format(key, value))
            fob.write('\t]\n')
        fob.write(']')
        return
    



