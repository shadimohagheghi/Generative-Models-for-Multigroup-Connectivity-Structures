import networkx as nx
from numpy import linalg as LA
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
import scipy.stats as st
plt.close("all")

##########################################################

def Bridging_ties_model(N_init, sampleN):
    clique_size = [3,4,5,6,7,8,9,10,11,12,13,14,15];      # clique sizes
    total1=0.0
    clique_num = np.zeros(len(clique_size))  # number of cliques of given size


    ##########################     Algorithm 1 Final Version      ######################

    xk = np.arange(13)
    pk = (1.0/27, 1.0/64, 1.0/125, 1.0/216, 1.0/343, 1.0/512, 1.0/729, 1.0/1000 , 1.0/1331, 1.0/1728, 1.0/2197, 1.0/2744, 1.0/3375)
    #pk = (1.0/9, 1.0/16, 1.0/25, 1.0/36, 1.0/49, 1.0/64, 1.0/81, 1.0/100 , 1.0/121, 1.0/144, 1.0/169, 1.0/196, 1.0/225)   # alpha= 2
    pk=np.multiply(13.3372634, pk)
    #pk=np.multiply(3.02627, pk)
    #print pk
    custm = st.rv_discrete(name='custm', values=(xk, pk))
    R = custm.rvs(size=N_init)
    R_sorted=sorted(R)
    counter=collections.Counter(R_sorted)

    keys=np.add(counter.keys(), 3)
    AClist= np.divide(counter.values(), keys)

    n_remaining= N_init-sum(np.multiply(AClist, keys))
    (q,r)=divmod(n_remaining,3)
    if n_remaining==5:
        AClist[2]=AClist[2]+1
    elif r==0:
        AClist[0]=AClist[0]+n_remaining/3
    else:
        if len(AClist)==1:
            AClist=np.append(AClist, 0)
        if (n_remaining % 3 == 2):
            AClist[0]=AClist[0]+n_remaining/3 - 2
            AClist[1]=AClist[1]+2
        if (n_remaining % 3 == 1):
            AClist[0]=AClist[0]+n_remaining/3 - 1
            AClist[1]=AClist[1]+1
    for i in range(len(AClist)):
        clique_num[i] = int(AClist[i])
    clique_num = map(int,clique_num)
    #print clique_num
    '''
    plt.figure(1)
    plt.scatter(counter.keys(),counter.values(),c='r',marker='o',s=100,alpha=0.5)
    plt.plot(counter.keys(),counter.values(),linewidth=2,c='r') 
    plt.xlabel('almost clique size',fontsize=10)
    plt.ylabel('number of almost cliques',fontsize=10)
    
    plt.figure(2)
    fig, ax = plt.subplots(1, 1)
    ax.plot(xk, custm.pmf(xk), 'ro', ms=12, mec='r')
    ax.vlines(xk, 0, custm.pmf(xk), colors='r', lw=4)
    plt.show()
    '''    
###########################################################
 
    def l_num(size):     # number of selected nodes given clique size
        return 1

###########################################################

    # almost cliques are generated
    Gclique,l,l_cliqeuwise,clique_sizes,rep_numbers_cliquewise,clique_ordered_cliquewise, clique_start  = almost_clique(clique_size,clique_num, l_num)
    #print(len(rep_numbers_cliquewise)), 'cliques formed'

####### MST of Randomly Weighted Undirected Complete Graph on n nodes ########

    GT = {}
    for u in range(len(rep_numbers_cliquewise)):
        GT[u] = {}
    for u in range(len(rep_numbers_cliquewise)):
        for v in range(u):
            r = random()
            GT[u][v] = r
            GT[v][u] = r
    T2 = mst(GT)
    mst_weight = sum([GT[u][v] for u,v in T2])

    T2G= nx.Graph()
    for u in range(len(rep_numbers_cliquewise)):
        for v in range(len(rep_numbers_cliquewise)):
            if (u,v) in T2:
                T2G.add_edge(u,v, weight=1)
    #print T2G.edges()
    plt.figure(1)
    nx.draw(T2G, pos=nx.spring_layout(T2G,k=0.1,iterations=200),node_size=500,cmap=plt.cm.Reds)
#############################################################################
                   
    for u in range(len(l)):  
        for v in range(len(l)):
            if u in T2G[v]:
                Gclique.add_edge(l[u],l[v])          
    
    #print sorted(nx.degree(Gclique).values())
    degreeList=sorted(nx.degree(Gclique).values())
    counter=collections.Counter(degreeList)
    
    plt.figure(2)
    
    plt.scatter(counter.keys(),counter.values(),c='r',marker='o',s=100,alpha=0.5)
    plt.plot(counter.keys(),counter.values(),linewidth=2,c='r') 
    plt.xlabel('node degree',fontsize=10)
    plt.ylabel('number of nodes',fontsize=10)
    plt.axis([0, 10, 0, 35])
    
    clique_list = []
    for i in range(len(clique_size)):
        dum =  np.linspace(clique_size[i],clique_size[i], clique_num[i])
        clique_list = np.hstack((clique_list,dum ))
    
    colors = []; c = 0
    for i in range(len(clique_list)):
        colors.extend(np.linspace(c,c,clique_list[i]))
        c = c + 1
 
    posx = []; posy = [];
    for i in range(len(clique_list)):
        centerx = np.cos(2*np.pi*i/len(clique_list)) 
        centery = np.sin(2*np.pi*i/len(clique_list))
        x1 = []; y1 = []; 
        for j in range(int(clique_list[i])):
            x1.append(centerx  + 0.2*np.cos(2*np.pi*j/clique_list[i]))
            y1.append(centery  + 0.2*np.sin(2*np.pi*j/clique_list[i]))
        posx.extend(x1); posy.extend(y1);
   
    pos = np.transpose(np.vstack((posx,posy)))
   
    plt.figure(3)
    nx.draw(Gclique, pos, node_color=colors,node_size=500,cmap=plt.cm.Reds)
    
    plt.show()
    A1= nx.to_numpy_matrix(Gclique)
    A = np.array(A1)
    A=np.trunc(A)

    #print "number of nodes: ", math.sqrt(np.size(A))
    
    np.savetxt('AdjMatrices/BT/BT'+str(N_init)+'_'+str(sampleN)+'.txt', A, fmt='%10.f', delimiter='\t') 
    '''
    with open('StructuralProperties/ClusteringCoeff/BT/BT'+str(N_init)+'.txt', "a") as text_file:
        text_file.write(str(nx.average_clustering(Gclique)) + "\n")

    with open('StructuralProperties/AveShortestPath/BT/BT'+str(N_init)+'.txt', "a") as text_file:
        text_file.write(str(nx.average_shortest_path_length(Gclique)) + "\n")

    with open('StructuralProperties/Density/BT/BT'+str(N_init)+'.txt', "a") as text_file:
        text_file.write(str(nx.density(Gclique)) + "\n")
    
    xdata,ydata = np.log10(counter.keys()),np.log10(counter.values())
    polycoef = np.polyfit(xdata, ydata, 1)
    yfit = 10**( polycoef[0]*xdata+polycoef[1] )

    with open('StructuralProperties/DegreeDist/BT/BT'+str(N_init)+'.txt', "a") as text_file:
        text_file.write(str(polycoef[0]) + "\n")
    
    #print polycoef[0]
    plt.subplot(211)
    plt.plot(xdata,ydata,'.k',xdata,yfit,'-r')
    plt.subplot(212)
    plt.loglog(xdata,ydata,'.k',xdata,yfit,'-r')
    plt.show()        
    '''
    
    return Gclique
    #return math.sqrt(np.size(A))
    
#Gclique = Bridging_ties_model(N_init, sampleN)
Gclique = Bridging_ties_model(14, 0)

