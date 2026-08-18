import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from netgraph import Graph

fig, ax = plt.subplots()
A = np.random.rand(5, 5)
G = nx.from_numpy_array(A, create_using=nx.DiGraph)
g = Graph(G, edge_cmap=plt.cm.viridis, ax=ax)
print(dir(g))
