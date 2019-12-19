from numpy import linalg as LA
import networkx as nx
from numpy import linalg as LA
import math
import csv
import collections
import numpy as np 
import matplotlib
from almost_clique import almost_clique 
from numpy import inf
from delta_ss import delta_ss
matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)
from random import random
from MinimumSpanningTree import MinimumSpanningTree as mst
import matplotlib.pyplot as plt
plt.close("all")

##########################################################

def FunctionalProperties(N_init, sampleN):

	A1 = np.loadtxt('AdjMatrices/BT/BT'+str(N_init)+'_'+str(sampleN)+'.txt')
	#A1=np.matrix([[0.1, 0.5, 0.4 ,0, 0], [ 0.1, 0.15 ,0.25, 0.25 ,0.25], [0.1, 0 ,0.3 ,0.3 ,0.3], [0.1, 0, 0.3, 0.3 ,0.3], [ 0.2, 0.2 ,0.2, 0.2 ,0.2]])		
	A2 = np.loadtxt('AdjMatrices/RT/RT'+str(N_init)+'_'+str(sampleN)+'.txt')
	A3 = np.loadtxt('AdjMatrices/C/C'+str(N_init)+'_'+str(sampleN)+'.txt')
	A4 = np.loadtxt('AdjMatrices/L/L'+str(N_init)+'_'+str(sampleN)+'.txt')

	n1=A1.shape[0]
	n2=A2.shape[0]
	n3=A3.shape[0]
	n4=A4.shape[0]

	##############################  Delta_ss  #############################
	
	dss1=delta_ss(A1)
	dss2=delta_ss(A2)
	dss3=delta_ss(A3)
	dss4=delta_ss(A4)

	#print dss1
	
	with open('CompAnalysis/EN1/SSMeanSqrEGTH/BT/BT_'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n1) + "\t")
		text_file.write(str('%.5f'%dss1) + "\n")
	
	with open('CompAnalysis/EN1/SSMeanSqrEGTH/RT/RT_'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n2) + "\t")
		text_file.write(str('%.5f'%dss2) + "\n")

	with open('CompAnalysis/EN1/SSMeanSqrEGTH/C/C_'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n3) + "\t")
		text_file.write(str('%.5f'%dss3) + "\n")
	with open('CompAnalysis/EN1/SSMeanSqrEGTH/L/L_'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n4) + "\t")	
		text_file.write(str('%.5f'%dss4) + "\n")
	
	
	##########################################################
	'''
	with open('StructuralProperties/TotalEdges/PYTHON/BT/'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n1) + "\t")
		text_file.write(str(A1.sum()/2) + "\n")

	w, v =LA.eig(A1)
	with open('CompAnalysis/SpectralRad/PYTHON/BT/'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n1) + "\t")
		text_file.write(str('%.5f'%np.max(w)) + "\n")
	rowsums=A1.sum(axis=1)
	D1_out=np.diag(rowsums, k=0)
	A1_EN1= np.dot(LA.inv(D1_out+np.identity(n1)),A1+np.identity(n1))
	v,w=LA.eig(A1_EN1)
	r_asym1=sorted(v)[n1-2]

	with open('CompAnalysis/EN1/ConvergenceFactor/PYTHON/BT/'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n1) + "\t")
   		text_file.write(str('%.5f'%r_asym1) + "\n")
   	
	##########################################################

	with open('StructuralProperties/TotalEdges/PYTHON/RT/'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n2) + "\t")
		text_file.write(str(A2.sum()/2) + "\n")

	w, v =LA.eig(A2)
	with open('CompAnalysis/SpectralRad/PYTHON/RT/'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n2) + "\t")
		text_file.write(str('%.5f'%np.max(w)) + "\n")

	rowsums=A2.sum(axis=1)
	D2_out=np.diag(rowsums, k=0)
	A2_EN1= np.dot(LA.inv(D2_out+np.identity(n2)),A2+np.identity(n2))
	v,w=LA.eig(A2_EN1)
	r_asym2=sorted(v)[n2-2]

	with open('CompAnalysis/EN1/ConvergenceFactor/PYTHON/RT/'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n2) + "\t")
		text_file.write(str('%.5f'%r_asym2) + "\n")
	
	##########################################################

	with open('StructuralProperties/TotalEdges/PYTHON/C/'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n3) + "\t")
		text_file.write(str(A3.sum()/2) + "\n")

	w, v =LA.eig(A3)
	with open('CompAnalysis/SpectralRad/PYTHON/C/'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n3) + "\t")
		text_file.write(str('%.5f'%np.max(w)) + "\n")

	rowsums=A3.sum(axis=1)
	D3_out=np.diag(rowsums, k=0)
	A3_EN1= np.dot(LA.inv(D3_out+np.identity(n3)),A3+np.identity(n3))
	v,w=LA.eig(A3_EN1)
	r_asym3=sorted(v)[n3-2]

	with open('CompAnalysis/EN1/ConvergenceFactor/PYTHON/C/'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n3) + "\t")
		text_file.write(str('%.5f'%r_asym3) + "\n")
	
	##########################################################
	
	with open('StructuralProperties/TotalEdges/PYTHON/L/'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str(n4) + "\t")
		text_file.write(str(A4.sum()/2) + "\n")

	w, v =LA.eig(A4)
	with open('CompAnalysis/SpectralRad/PYTHON/L/'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str('%.5f'%n4) + "\t")
		text_file.write(str('%.5f'%np.max(w)) + "\n")
	rowsums=A4.sum(axis=1)
	D4_out=np.diag(rowsums, k=0)
	A4_EN1= np.dot(LA.inv(D4_out+np.identity(n4)),A4+np.identity(n4))
	v,w=LA.eig(A4_EN1)
	r_asym4=sorted(v)[n4-2]

	with open('CompAnalysis/EN1/ConvergenceFactor/PYTHON/L/'+str(N_init)+'.txt', "a") as text_file:
		text_file.write(str('%.5f'%n4) + "\t")
  		text_file.write(str('%.5f'%r_asym4) + "\n")
	'''
	
#FunctionalProperties(30, 0)

