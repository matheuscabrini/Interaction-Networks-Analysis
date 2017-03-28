import numpy as np
import networkx as nx
import pylab as plt
from mynetlibs.measures import NetworkMeasures
from networkx.algorithms import bipartite

# Prints the chosen measures of network G
def get_measures(G):
    nm = NetworkMeasures(G)
    print('Average betweenness = ', nm.betweenness())
    print('Average clustering coefficient = ', nm.clustering())
    print('Transitivity = ', nm.transitivity())
    print('Assortativity = ', nm.assortativity())
    print('Average shortest path length - ', nm.avg_shortest_path_len())
    print('2nd moment of degree distribution = ', nm.moment_degree_distrib(2)) 
    print('Shannon Entropy of degree distribution = ', nm.shannon_entropy()) 
    nm.degree_distrib_plot()

# Getting input network (adjacency matrix) from txt
file_name = "bromelias.txt"
network_name = '.'.join(file_name.split('.')[:-1]) if '.' in file_name else file_name
B = np.loadtxt(file_name, dtype=int)
is_bp = False

try:
    G = nx.Graph(B) 
except nx.exception.NetworkXError: # if matrix is not square (bipartite), exception will occur 
    is_bp = True
else:
    if bipartite.is_bipartite(G) is True:
        is_bp = True

if is_bp is False:
    """ UNIPARTITE NETWORK """

    # Getting the measures and drawing the network
    G = nx.Graph(B)
    get_measures(G)
    nx.draw(G)
    plt.savefig("graph.png")
    plt.show()

else:
    """ BIPARTITE NETWORK """

    # Constructing graph, adding and labeling nodes
    G = nx.Graph()
    a = ['a'+str(i) for i in range(B.shape[0])]
    b = ['b'+str(j) for j in range(B.shape[1])]
    G.add_nodes_from(a, bipartite=0)
    G.add_nodes_from(b, bipartite=1)

    # Adding edges
    r, c = np.where(B != 0)
    for i, j in zip(r, c):
        G.add_edge(a[i], b[j])

    # Drawing bipartite network
    r_set, c_set = bipartite.sets(G)
    pos = dict()
    pos.update( (n, (1, i)) for i, n in enumerate(r_set) ) # put nodes from r_set at x=1
    pos.update( (n, (2, i)) for i, n in enumerate(c_set) ) # put nodes from c_set at x=2
    #plt.title("Bipartite graph")
    (plt.gcf()).canvas.set_window_title('Bipartite graph')
    nx.draw(G, pos=pos, with_labels=True)
    nx.draw_networkx_nodes(G, pos, nodelist=r_set, node_color='r', node_size=500, alpha=1.0)
    nx.draw_networkx_nodes(G, pos, nodelist=c_set, node_color='b', node_size=500, alpha=0.8)
    plt.savefig(network_name + '-bipartite-graph.png')
    plt.show()

    # Drawing rows set projection
    Pr = bipartite.weighted_projected_graph(G, list(r_set))
    plt.title('Rows Set Projection')
    (plt.gcf()).canvas.set_window_title('Rows Set Projection')
    pos=nx.spring_layout(Pr)
    nx.draw(Pr,pos,with_labels=True)
    labels = nx.get_edge_attributes(Pr,'weight')
    nx.draw_networkx_edge_labels(Pr,pos,edge_labels=labels)
    plt.savefig(network_name + '-row-projection-graph.png')
    plt.show()

    # Drawing columns set projection 
    Pc = bipartite.weighted_projected_graph(G, list(c_set))
    plt.title('Columns Set Projection')
    (plt.gcf()).canvas.set_window_title('Columns Set Projection')
    pos=nx.spring_layout(Pc)
    nx.draw(Pc, with_labels=True)
    labels = nx.get_edge_attributes(Pc,'weight')
    #nx.draw_networkx_edge_labels(Pc,pos,edge_labels=labels)
    plt.savefig(network_name + '-col-projection-graph.png')
    plt.show()

    # Getting measures from the row and col projections
    print("\nROW SET PROJECTION MEASURES")
    get_measures(Pr)
    print("\nCOLUMN SET PROJECTION MEASURES")
    get_measures(Pc)

""" # Passando nome de arquivo como parametro
fname = input()
with open(fname, 'r') as f:
    count = 0 
    read_data = f.readline()
    while(read_data):
        print(len(read_data))
        count += 1
        read_data = f.readline()
    print("count ", count)
# B = np.genfromtxt(fname)
"""
