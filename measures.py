"""
http://arxiv.org/pdf/1511.04453v1.pdf --ARTIGO
http://www.nature.com/nature/journal/v410/n6825/fig_tab/410268a0_F7.html -- bptite projection
https://github.com/dgorissen/pycel --xls to txt?
https://pypi.python.org/pypi/pyexcel-xls/0.1.0 --lucas passou
https://github.com/grupy-sanca/dojo-toolkit/blob/master/dojo_toolkit/dojo.py --PYTHON
https://networkx.github.io/documentation/networkx-1.9.1/reference/readwrite.html --lista de interfaces R/W c nx
http://www.mcz.harvard.edu/Departments/PopGenetics/pdf/2005_01_-_Robustness_and_network.pdf --network entropy
https://www.dropbox.com/s/ylv9xan68sc5m2v/book_chapter_10.pdf?dl=0 --barabasi
https://networkx.github.io/documentation/networkx-1.9/examples/drawing/degree_histogram.html --GRAFICO
https://www.nceas.ucsb.edu/interactionweb/resources.html  --MATRIZES

Calcular as medidas: Tabela onde cada linha é uma rede e cada coluna, uma medida.
Medidas:
average clustering coefficient, transitivity, assortativity, average shortest path length,
average betweenness centrality, Shannon Entropy, Second moment of the degree distribution, 
Gráficos: degree distribution (in degree and out degree).

TODO: 
-teste de mesa com algo tipo >>> B.add_edges_from([('a', 1), ('b', 1), ('a', 2), ('b', 2)])
e ver como fica a projecao do bipartido. 
    https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.bipartite.html
-grafico degree distrib
-colocar measures numa funcao, e chamar pra {Pr, Pc, G} em bp, senao só em G (mas prolly virá apenas grafo bp)
-tabela
-modularizacao & GITHUB?
"""

import numpy as np
import networkx as nx
import pylab as plt
from networkx.algorithms import bipartite

B = np.genfromtxt("dupont.txt")
bp = True

if bp is False:
    G = nx.Graph(B)

    # Drawing graph
    nx.draw(G)
    plt.savefig("graph.png")
    plt.show()

else:
    """ Bipartite processing: 
    The adjacency matrix A of a bipartite graph whose two parts have sets r and c of vertices can be written in the form 
    A = [ 0{r x r},    B        ] 
        [ B^T,         0{c x c} ]
    where B is an r × c matrix, and 0{r x r} and 0{c x c} represent the r × r and c × c zero matrices. 
    In this case, the smaller matrix B uniquely represents the graph, and the remaining parts of A 
    can be discarded as redundant. B is sometimes called the biadjacency matrix.
    """  
    rows, cols = B.shape
    Atop = np.concatenate((np.zeros((rows, rows)), B), axis=1)
    Adown = np.concatenate((np.transpose(B), np.zeros((cols, cols))), axis=1)
    A = np.concatenate((Atop, Adown), axis=0)
    np.savetxt('bp.txt', A, fmt='%u')

    G = nx.Graph(A)
    r_set, c_set = bipartite.sets(G)
    pos = dict()
    pos.update( (n, (1, i)) for i, n in enumerate(r_set) ) # put nodes from r_set at x=1
    pos.update( (n, (2, i)) for i, n in enumerate(c_set) ) # put nodes from c_set at x=2

    nx.draw(G, pos=pos)
    nx.draw_networkx_nodes(G, pos, nodelist=r_set, node_color='r', node_size=500, alpha=1.0)
    nx.draw_networkx_nodes(G, pos, nodelist=c_set, node_color='b', node_size=500, alpha=0.8)

    # Drawing bipartite graph
    plt.savefig("graph.png")
    plt.show()

    # Drawing biadjacency rows set projection 
    Pr = bipartite.projected_graph(G, r_set, multigraph=False)
    nx.draw(G)
    plt.savefig("graph.png")
    plt.show()

    # Drawing biadjacency columns set projection 
    Pc = bipartite.projected_graph(G, c_set, multigraph=False)
    nx.draw(G)
    plt.savefig("graph.png")
    plt.show()

""" MEASURES """

betw_list = []
betw = nx.betweenness_centrality(G)
for node_betw in betw.items():
    node, betw = node_betw[0], node_betw[1]
    betw_list.append(node_betw[1])
    #print("Node {:d} has betweenness {:f}".format(node, betw))
avg_betweenness = np.mean(betw_list)
print("Average betweenness - ", avg_betweenness)

avg_cl_coef = nx.average_clustering(G)
print("Average clustering coefficient - ", avg_cl_coef)

trans = nx.transitivity(G)
print("Transitivity - ", trans)

assort = nx.degree_assortativity_coefficient(G)
print("Assortativity - ", assort)

avg_shortest_path_len = nx.average_shortest_path_length(G)
print("Average shortest path length - ", avg_shortest_path_len)


""" 2nd moment of degree distribution """

# degree distribution is defined as Pk = Nk/N 
# Pk is the probability of a randomly chosen node having degree k
# Nk is the number of nodes which have degree k, and N is the total of nodes in graph 

degrees = list(nx.degree(G).values())
n = [0] * (1 + max(degrees))
for k in degrees:
    n[k] += 1

p = [0] * (1 + max(degrees))
n_nodes = nx.number_of_nodes(G)
for k in range(0, 1 + max(degrees)):
    p[k] = n[k] / n_nodes 
    print("P(node having degree {:d}) = {:f}".format(k, p[k])) 

# nth moment of distribution is:
# m = (x1 ^ n + x2 ^ n + ... + xN ^ n) / N
n = 2
m = 0
for x in p:
    m += pow(x, n) 
m = m/len(p)
print("2nd moment of degree distribution: m = ", m) 


""" Shannon Entropy of Degree Distribution """
# H = - sum in j (Pij . log(pij))
h = 0
for k in range(0, 1 + max(degrees)):
    if p[k] != 0:
        h -= p[k] * np.log10(p[k])
print("Shannon's Entropy of degree distribution: h = ", h)
 

""" Degree Distribution plot ( k x p[k] ) """ 

plt.plot(list(range(0, 1+max(degrees))), p, 'ro')
plt.axis([0, 18, 0, 1])
plt.show()


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


