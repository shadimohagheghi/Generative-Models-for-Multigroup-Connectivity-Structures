import networkx as nx
import math
import collections
import numpy as np 
import matplotlib
from almost_clique import almost_clique 
from numpy import inf
matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)
from random import random
from MinimumSpanningTree import MinimumSpanningTree as mst
import matplotlib.pyplot as plt
plt.close("all")

G = nx.scale_free_graph(100,)

plt.figure()
nx.draw(G, pos=nx.circular_layout(G))

degreeList=sorted(nx.degree(G).values())
counter=collections.Counter(degreeList)
xdata,ydata = np.log10(counter.keys()),np.log10(counter.values())
polycoef = np.polyfit(xdata, ydata, 1)
yfit = 10**( polycoef[0]*xdata+polycoef[1] )

plt.figure()
print polycoef[0]
plt.subplot(211)
plt.plot(xdata,ydata,'.k',xdata,yfit,'-r')
plt.subplot(212)
plt.loglog(xdata,ydata,'.k',xdata,yfit,'-r')
plt.show()


plt.show()