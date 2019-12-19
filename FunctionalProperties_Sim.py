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
i=0

from FunctionalProperties import FunctionalProperties

for n in range(96, 97):
	for i in range(1):
		print i
		FunctionalProperties(n,i)
	i=0
