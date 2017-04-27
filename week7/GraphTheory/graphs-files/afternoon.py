import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

G = nx.read_edgelist('data/imdb_edges.tsv', delimiter='\t')

G1 = nx.read_edgelist('data/actor_edges.tsv', delimiter='\t')

nodes = G1.degree().values()
plt.hist(nodes,bins=25)
plt.xlim(0,200)
plt.show()

Counter(nx.degree_centrality(G)).most_common(5)

len(list(nx.connected_components(G1)))


size = [len(c) for c in nx.connected_components(G1)]

plt.hist(size[1:])

G2 = nx.read_edgelist('data/small_actor_edges.tsv', delimiter='\t')

len(list(nx.connected_components(G2)))

Counter(nx.degree_centrality(G2)).most_common(5)

Counter(nx.betweenness_centrality(G2)).most_common(5)


karateG = nx.karate_club_graph()

# betweenness= nx.edge_betweenness_centrality(karateG)
#
# u,v = sorted(betweenness.items(), key=lambda x: x[1])[-1][0]
#
# karateG.remove_edge(u,v)
