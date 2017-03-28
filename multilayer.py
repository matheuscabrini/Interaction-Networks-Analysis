import numpy as np
import networkx as nx
import pylab as plt
from networkx.algorithms import bipartite
from mynetlibs.measures import NetworkMeasures

"""
todo: try these libs
https://bitbucket.org/bolozna/multilayer-networks-library/
https://github.com/nkoub/multinetx
"""

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

# Getting input networks (adjacency matrices) from the txts
mG1 = np.genfromtxt("multilayertest1.txt")
mG2 = np.genfromtxt("multilayertest2.txt")
eps = 0.5 # interlayer edges coefficient for the identity matrices

""" Multiplex graph processing: 
The adjacency matrix M of a multiplex graph with subgraphs G1 and G2 can be written in the form 
A = [ G1,       eps*I ] 
    [ eps*I,    G2    ]
where I is the identity matrix, and eps is the interlayer edges coefficient (0 < eps < 1). 
""" 
rowsG1, colsG1 = mG1.shape
rowsG2, colsG2 = mG2.shape
Mtop = np.concatenate((mG1, eps*np.identity(rowsG1)), axis=1)
Mdown = np.concatenate((eps*np.identity(rowsG2), mG2), axis=1)
M = np.concatenate((Mtop, Mdown), axis=0)
np.savetxt('multilayertestresult.txt', M, fmt='%.2f', delimiter='  ')

# Now, we get construct the network G from matrix M and extract it's measures
G = nx.Graph(M)
get_measures(G)

# Drawing graph
plt.title('Multilayer network')
pos=nx.spring_layout(G)
nx.draw(G,pos,with_labels=True)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plt.savefig("multilayernet.png")
plt.show()



