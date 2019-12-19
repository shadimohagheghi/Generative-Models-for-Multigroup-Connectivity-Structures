
from numpy import linalg as LA
import networkx as nx
import math
import csv
import collections
import operator
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

def delta_ss(P):

	n = P.shape[0]

	ones_v=np.ones(n);
	Ones_m=np.ones((n,n))

	w=np.zeros(n)
	pi=np.zeros(n)
	Z=np.zeros((n,n))
	D=np.zeros((n,n))

	D1_out=np.diag(P.sum(axis=1))

	P= np.dot ( LA.inv(D1_out + np.diag(ones_v) ), (P + np.diag(ones_v)))

	[v,w]=LA.eig(np.transpose(P))


	ind = np.unravel_index(np.argmax(v, axis=None), v.shape)
	print np.argmax(v)

	pi = [row[np.argmax(v)] for row in w]
	pi=np.divide(pi, sum(pi))
	
	Z = LA.inv ( (np.diag(ones_v)) - P + np.outer(ones_v,np.transpose(pi)) )
	D=LA.inv(np.diag(pi));


	M=np.dot( (np.diag(ones_v) - Z + np.dot(Ones_m, np.diag(np.diag(Z))) )  , D)
	for i in range(n):
		for j in range(n):
			if np.abs(M[i][j])<0.000001:
				M[i][j]=0

	dss= np.dot(np.dot(np.transpose(pi),   np.dot(M, (np.dot(np.diag(pi), np.diag(pi) ) ) )   ), ones_v)
	return dss

	'''
	n1=A1.shape[0]
	e=np.ones((n1,1))
	et= np.ones(n1)
	S=np.ones(n1);
	E=np.ones((n1,n1))
	mu=np.zeros((n1,n1))
	A=np.zeros((n1,n1))
	PP=np.zeros((n1,n1))
	mu[:,n1-1]=1	
	P_new=np.zeros((n1,n1))
	M=np.zeros((n1,n1));
	M_EGTH=np.zeros((n1,n1));	
	for i in range(n1):
		for j in range(n1):
		    A[i,j]=A1[i,j]
		    PP[i,j]=A1[i,j]

	for k in range(n1):
		for n in range(n1-1,0,-1):
			temp_sum=0
			for z in range(n):
				temp_sum=temp_sum+PP[n,z]
			pp_sum=PP.sum(axis=1)
			S[n]=temp_sum
			for i in range(n):
				for j in range(n):
					PP[i,j]=PP[i,j]+PP[i,n]*PP[n,j]/S[n]
		    		mu[i,n-1]=mu[i,n]+mu[n,n]*PP[i,n]/S[n]
		M[0,k]=((PP[1,0]*mu[0,1])+(PP[0,1]*mu[1,1]))/PP[1,0]
		for n in range(1,n1):
		    mm=0;
		    for i in range(1, n):
		        mm=mm+PP[n,i]*M[i,k]
		    M[n,k]=(mm+mu[n,n])/S[n]
		for col in range(n1):
		    for row in range(n1):
		        P_new[np.remainder(row+1+n1-2,n1), np.remainder(col+1+n1-2,n1)]=A[row,col]

		for i in range(n1):
		    for j in range(n1):
		        A[i,j]=P_new[i,j]
		        PP[i,j]=A[i,j]
	for col in range(n1):
		for row in range(n1):
		    M_EGTH[np.remainder(row+1+col+1-2,n1),col]=M[row,col]
	#print "M_EGTH", M_EGTH
	w=np.zeros((n1,1))
	m=A1.sum()/2;
	d_out=A1.sum(axis=1)
	w=(d_out+1)/(2*m+n1);
	#print "np.transpose(w)",np.transpose(w)
	d_ss= np.dot(np.dot(np.transpose(w),M_EGTH),np.dot(np.diag(w),np.diag(w)))
	#print "delta_ss", d_ss
	delta_ss=np.dot(d_ss, np.ones((n1,1)))
	#print np.dot(d_ss, np.ones((n1,1)))

	return delta_ss
	'''

