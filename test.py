import networkx as nx
import matplotlib.pyplot as plt


# draw example
# graph is a list of tuples of nodes. Each tuple defining the
# connection between 2 nodes
graph= nx.complete_graph(8)
nx.draw(graph,pos=nx.spring_layout(graph))
plt.show()
#draw_graph(graph)