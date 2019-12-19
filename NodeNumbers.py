import networkx as nx
from allfunctions import draw_degdist
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

from Liaison_model3 import Liaison_model3
from Bridging_ties_model import Bridging_ties_model
i=0

for n in range(30,100,1):
	print Bridging_ties_model(n,0)
'''
for n in range(30,530,10):
	for i in range(10):
		print Bridging_ties_model(n,i)
	i=0
'''

