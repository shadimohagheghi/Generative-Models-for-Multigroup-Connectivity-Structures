import networkx as nx
import math
import collections
import numpy as np 
import matplotlib
from almost_clique import almost_clique 
from numpy import inf
from operator import add
matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)
from random import random
#from RandomTree_23 import RandomTree_23
from MinimumSpanningTree import MinimumSpanningTree as mst
import matplotlib.pyplot as plt
import scipy.stats as st
plt.close("all")

##########################################################

def Liaison_model_final(N_init, sampleN):
    clique_size = [3,4,5,6,7,8,9,10,11,12,13,14,15];      # clique sizes
    total1=0.0
    clique_num = np.zeros(len(clique_size))  # number of cliques of given size

    ##########################     Algorithm 1 Final Version     ######################

    xk = np.arange(13)
    pk = (1.0/27, 1.0/64, 1.0/125, 1.0/216, 1.0/343, 1.0/512, 1.0/729, 1.0/1000 , 1.0/1331, 1.0/1728, 1.0/2197, 1.0/2744, 1.0/3375)
    #pk = (1.0/9, 1.0/16, 1.0/25, 1.0/36, 1.0/49, 1.0/64, 1.0/81, 1.0/100 , 1.0/121, 1.0/144, 1.0/169, 1.0/196, 1.0/225)
    pk=np.multiply(13.3372634, pk)
    #pk=np.multiply(3.02627, pk)
    custm = st.rv_discrete(name='custm', values=(xk, pk))
    R = custm.rvs(size=N_init)

    degreeList=sorted(R)
    counter=collections.Counter(degreeList)

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
 
    ###########################################################
 
    def l_num(size):     # number of leaders given clique size
        return 1

    ###########################################################

    # almost cliques are generated
    Gclique,l,l_cliqeuwise,clique_sizes,rep_numbers_cliquewise,clique_ordered_cliquewise, clique_start  = almost_clique(clique_size,clique_num, l_num)
    #print(len(rep_numbers_cliquewise)), 'cliques formed'
    N_AC=len(rep_numbers_cliquewise)

    ##########################################################
    
    xk = np.arange(2)
    pk = (1.0/8, 1.0/27)
    pk=np.multiply(6.17143, pk)
    custm = st.rv_discrete(name='custm', values=(xk, pk))
    R = custm.rvs(size=N_AC)

    degreeList=sorted(R)
    counter=collections.Counter(degreeList)

    keys=np.add(counter.keys(), 2)
    AClist= np.divide(counter.values(), keys)
    plist=np.zeros(len(pk))
    for i in range(len(AClist)):
        plist[i] = AClist[i]

    n_remaining= N_AC-sum(np.multiply(AClist, keys))

    clique_list = []
    for i in range(len(clique_size)):
        dum =  np.linspace(clique_size[i],clique_size[i], clique_num[i])
        clique_list = np.hstack((clique_list,dum ))

    parents_4_list=[]
    parents_3_list=[]
    parents_2_list=[]

    n= n_remaining
    if len(AClist)==1:
        AClist=np.append(AClist, 0) 

    if (n_remaining%2==0):
        AClist[0]=AClist[0]+n_remaining/2
    elif (n_remaining==1):
        if AClist[1]>0:
            AClist[1]=AClist[1]-1
            AClist[0]=AClist[0]+2
        elif AClist[1]==0:
            AClist[1]=AClist[1]+1
            AClist[0]=AClist[0]-1
    elif (n_remaining%3==0):
        AClist[1]=AClist[1]+n_remaining/3

    n=sum(AClist)

    if len(AClist)==1:
        AClist[1]=0

    ####################################################################################

    l_3_list=[]
    l_2_list=[]
    l_list=[]
    prev_l=AClist[0]+AClist[1]

    while prev_l>5:

        xk = np.arange(2)
        pk = (1.0/8, 1.0/27)
        pk=np.multiply(6.17143, pk)
        custm = st.rv_discrete(name='custm', values=(xk, pk))
        R = custm.rvs(size=prev_l)

        degreeList=sorted(R)
        counter=collections.Counter(degreeList)

        keys=np.add(counter.keys(), 2)
        l_list= np.divide(counter.values(), keys)
        n_remaining= prev_l-sum(np.multiply(l_list, keys))

        if len(l_list)==1:
            l_list=np.append(l_list, 0) 

        if (n_remaining%2==0):
            l_list[0]=l_list[0]+n_remaining/2
        elif (n_remaining==1):
            if l_list[1]>0:
                l_list[1]=l_list[1]-1
                l_list[0]=l_list[0]+2
            elif l_list[1]==0:
                l_list[1]=l_list[1]+1
                l_list[0]=l_list[0]-1
        elif (n_remaining%3==0):
            l_list[1]=l_list[1]+n_remaining/3

        l_2_list.append(l_list[0])
        l_3_list.append(l_list[1])

        prev_l=sum(l_list)

        if len(l_list)==1:
            l_list[1]=0

    if prev_l==3:
        l_3_list.append(1)
        l_2_list.append(0)
    if prev_l==4:
        l_2_list.append(2)
        l_3_list.append(0)

        l_2_list.append(1)
        l_3_list.append(0)
    if prev_l==5:
        l_2_list.append(1)
        l_3_list.append(1)
        l_2_list.append(1)
        l_3_list.append(0)
    elif prev_l==2:
        l_3_list.append(0)
        l_2_list.append(1)
 
    l_2_list.insert(0,AClist[0])
    l_3_list.insert(0,AClist[1])

    l_list=map(add, l_2_list, l_3_list)

    i=0
    j=0

    n1=len(Gclique.nodes())
    L=nx.empty_graph(int(len(l_list)))
    Gclique.add_nodes_from(L.nodes())
    Gclique=nx.convert_node_labels_to_integers(Gclique,first_label=0)

    levels=len(l_list)
    n=[]
    n.append(n1)
    for i in range(1, int(levels)+1):
        n.append(n[i-1]+l_list[i-1])
    ####################################################################################
    
    j=0
    for i in clique_start:
        if clique_start.index(i)<2*(l_2_list[0]):
            Gclique.add_edge(i, n[0]+int(clique_start.index(i)/2))

    n3_start=clique_start[2*l_2_list[0]-1]
    n3_end=clique_start[len(clique_start)-1]

    if (clique_start.index(n3_end)-clique_start.index(n3_start)>0):
        for i in range(clique_start.index(n3_start)+1, clique_start.index(n3_end)+1):
            Gclique.add_edge(clique_start[i], n[0]+ int(clique_start.index(n3_start)/2)+1 + int((i - clique_start.index(n3_start)-1)/3)) 

    for j in range(1, int(levels)):
        for i in range(2*l_2_list[j]):
            Gclique.add_edge(i+n[j-1], n[j]+int(i/2))

        for i in range(3*l_3_list[j]):
            Gclique.add_edge(2*l_2_list[j]+i+n[j-1], l_2_list[j]+n[j]+int(i/3))

#############################################################################
    degreeList=sorted(nx.degree(Gclique).values())
    counter=collections.Counter(degreeList)
    
    plt.figure(1)
    
    plt.scatter(counter.keys(),counter.values(),c='r',marker='o',s=100,alpha=0.5)
    plt.plot(counter.keys(),counter.values(),linewidth=2,c='r') 
    plt.xlabel('node degree',fontsize=10)
    plt.ylabel('number of nodes',fontsize=10)
    plt.axis([0, 10, 0, 35])
    
    colors = []; c = 0
    for i in range(len(clique_list)):
        colors.extend(np.linspace(c,c,clique_list[i]))
        c = c + 1
 
    posx = []; posy = [];
    for i in range(len(clique_list)):
        centerx = 0.5*i/len(clique_list)
        centery = 0.0075/len(clique_list)
        x1 = []; y1 = []; 
        for j in range(int(clique_list[i])):
            x1.append(centerx  + 0.005*np.cos(2*np.pi*j/clique_list[i]))
            y1.append(centery  + 0.0005*np.sin(2*np.pi*j/clique_list[i]))
        posx.extend(x1); posy.extend(y1);

    x1 = []
    y1 = []
    i=0 
    for i in range(int(levels)):
        #print "i",i
        for j in range(int(l_list[i])):
            #print "j", j
            x1.append(0.5*j/int(l_list[i]))
            y1.append(((i+2.5)*0.02)/levels)
    posx.extend(x1); posy.extend(y1); 
    pos = np.transpose(np.vstack((posx,posy)))
    print "len(pos) ", len(pos)

    plt.figure(2)
    #nx.draw(Gclique,pos,node_color=colors,node_size=50,cmap=plt.cm.Reds)
    nx.draw(Gclique, pos, node_size=50,cmap=plt.cm.Reds)
    
    plt.show()
    '''
    A1= nx.to_numpy_matrix(Gclique)
    A = np.array(A1)
    A=np.trunc(A)
    #print A.shape[0]
    
    np.savetxt('AdjMatrices/L/L'+str(N_init)+'_'+str(sampleN)+'.txt', A, fmt='%10.f', delimiter='\t') 

    with open('StructuralProperties/ClusteringCoeff/L/L'+str(N_init)+'.txt', "a") as text_file:
        text_file.write(str(nx.average_clustering(Gclique)) + "\n")

    with open('StructuralProperties/AveShortestPath/L/L'+str(N_init)+'.txt', "a") as text_file:
        text_file.write(str(nx.average_shortest_path_length(Gclique)) + "\n")

    with open('StructuralProperties/Density/L/L'+str(N_init)+'.txt', "a") as text_file:
        text_file.write(str(nx.density(Gclique)) + "\n")
    
    xdata,ydata = np.log10(counter.keys()),np.log10(counter.values())
    polycoef = np.polyfit(xdata, ydata, 1)
    yfit = 10**( polycoef[0]*xdata+polycoef[1] )

    with open('StructuralProperties/DegreeDist/L/new/L'+str(N_init)+'.txt', "a") as text_file:
        text_file.write(str(polycoef[0]) + "\n")
    
    print polycoef[0]
    plt.subplot(211)
    plt.plot(xdata,ydata,'.k',xdata,yfit,'-r')
    plt.subplot(212)
    plt.loglog(xdata,ydata,'.k',xdata,yfit,'-r')
    plt.show()        
    '''
    return Gclique
    
#Gclique = Liaison_model_final(N_init, sampleN)
Gclique = Liaison_model_final(50, 0)

