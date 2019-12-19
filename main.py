import networkx as nx
import numpy as np 
import matplotlib
import time
from almost_clique import almost_clique 
from numpy import inf
import subprocess, os
matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)
import matplotlib.pyplot as plt

plt.close("all")

from Liaison_model_final import Liaison_model_final
from Bridging_ties_model import Bridging_ties_model
from comembership_model2 import comembership_model2
from redundant_ties_model2 import redundant_ties_model2

i=0


for n in range(775,785,5):
	for i in range(100):
		
		Bridging_ties_model(n,i)
		redundant_ties_model2(n,i)
		comembership_model2(n,i)
		Liaison_model_final(n,i)
	i=0

#for n in range(2500, 10000, 500):
#	for i in range(10):
#		internal_liaison_model(n,i)
#		redundant_ties_model(n,i)
#		comembership_model(n,i)
#		external_liaison_model(n,i)
#	i=0

