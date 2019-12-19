import networkx as nx
from allfunctions import draw_degdist
import numpy as np 
import matplotlib
import time
from almost_clique import almost_clique 
from numpy import inf
import math
import subprocess, os
matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)
import matplotlib.pyplot as plt

plt.close("all")

#from Liaison_model4 import Liaison_model4
#from Bridging_ties_model import Bridging_ties_model
#from comembership_model2 import comembership_model2
#from redundant_ties_model2 import redundant_ties_model2

def RandomTree_23(N_AC):
	T_Graph= nx.Graph()
	#T_Graph.add_node(N_AC)
	T_list=[]
	parents_3=0
	parents_2=0
	parents=0
	n= N_AC
	while n>1:
		(q,r)=divmod(n,3)
		print "(q,r): ", (q,r)
		if r==1:
			parents_3=parents_3+q-1
			parents_2=parents_2+2
		else:
			parents_3=parents_3+q
			if r==2:
				parents_2=parents_2+1
		l=math.ceil(n/3.0)
		print "l: ", l
		n=l
		parents=parents+l	
	print "parents_3: ", parents_3
	print "parents_2: ", parents_2
	print "parents: ", parents
	print "levels: ", math.ceil(math.log(N_AC)/math.log(3))

	#plt.figure()
	#nx.draw(T_Graph, pos=nx.spectral_layout(T_Graph)) 
	#plt.show()



RandomTree_23(30)

