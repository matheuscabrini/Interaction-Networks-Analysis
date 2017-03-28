import numpy as np
import networkx as nx
import pylab as plt

class NetworkMeasures:

	def __init__(self, G):
		# Let's get the degree distribution p, which will be used for some measures. 
		# It is defined as Pk = Nk/N (probability of a randomly chosen node having degree k)
		# Nk is the number of nodes which have degree k, and N is the total of nodes in graph
		self.G = G 
		self.degrees = list(nx.degree(G).values())
		# n = [0] * (1 + max(degrees)) shared references causes problems
		n = [0 for row in range(1 + max(self.degrees))] # list comprehension works better
		for k in self.degrees: 
		    n[k] += 1
		self.degree_distrib = [0 for row in range(1 + max(self.degrees))] 
		n_nodes = nx.number_of_nodes(G)
		for k in range(0, 1 + max(self.degrees)):
		    self.degree_distrib[k] = n[k] / n_nodes 
		    print("P(node having degree {:d}) = {:f}".format(k, self.degree_distrib[k]))

	def betweenness(self):
		betw_list = []
		betw = nx.betweenness_centrality(self.G)
		for node_betw in betw.items():
		    node, betw = node_betw[0], node_betw[1]
		    betw_list.append(node_betw[1])
		    #print("Node {:d} has betweenness {:f}".format(node, betw))
		return np.mean(betw_list)

	def clustering(self):
		return nx.average_clustering(self.G)

	def transitivity(self):
		return nx.transitivity(self.G)

	def assortativity(self):
		return nx.degree_assortativity_coefficient(self.G)

	def avg_shortest_path_len(self):
		if nx.is_connected(self.G) is True:
			return nx.average_shortest_path_length(self.G)
		else:
		    return 'Cannot calculate. Graph is not connected'

	""" 2nd moment of degree distribution """
	def moment_degree_distrib(self, k):
		# kth moment of distribution is:
		# m = (x1 ^ k + x2 ^ k + ... + xN ^ k) / N
		m = 0
		for x in self.degree_distrib:
		    m += pow(x, k) 
		m = m/len(self.degree_distrib)
		return m

	""" Shannon Entropy of Degree Distribution """
	# H = - sum in j (Pij . log(pij))
	def shannon_entropy(self): 
		h = 0
		for k in range(0, 1 + max(self.degrees)):
		    if self.degree_distrib[k] != 0:
		        h -= self.degree_distrib[k] * np.log10(self.degree_distrib[k])
		return h
	 
	""" Degree Distribution plot ( k x p[k] ) """ 
	# if the graph is not bipartite, it can be directed
	# therefore, we may have to plot in and out degree distribs
	def degree_distrib_plot(self):
		plt.title('Degree Distribution')
		plt.xlabel('degree k')
		plt.ylabel('p (k)')	
		plt.plot(list(range(0, 1+max(self.degrees))), self.degree_distrib, 'ro')
		plt.axis([0, 18, 0, 1])
		plt.show()

