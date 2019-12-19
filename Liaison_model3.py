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

##########################################################

def Liaison_model3(N_init, sampleN):
    clique_size = [3,4,5,6,7,8,9,10,11,12,13,14,15];      # clique sizes
    clique_edges=np.asarray(clique_size)
    clique_edges = np.divide(np.multiply(clique_edges,clique_edges+1),2)
    ep = 0.1

    total1=0
    clique_num = np.zeros(len(clique_size))  # number of cliques of given size
    for w in range(100):
        for i in range(len(clique_size)):
            clique_num[i] = clique_num[i]+(2.5321*(float(N_init-total1)/(clique_size[i])**3));
        clique_num = map(int, clique_num)
        total1= sum(np.multiply(clique_size, clique_num))
        #print total1 
    # comment this next line to generate a small sample graph easy to visualize
    #clique_num = [0,7,3,1,0,0,0,0]
    #6/(np.pi)^2
 
    ###########################################################
 
    def l_num(size):     # number of leaders given clique size
        return 1

    ###########################################################

    # almost cliques are generated
    Gclique,l,l_cliqeuwise,clique_sizes,rep_numbers_cliquewise,clique_ordered_cliquewise, clique_start  = almost_clique(clique_size,clique_num, l_num)
    #print(len(rep_numbers_cliquewise)), 'cliques formed'

    ################### Spanning Tree with Power Law #####################
    '''
    global T
    T=nx.random_powerlaw_tree(len(rep_numbers_cliquewise), tries=1000000) 

    plt.figure(2)
    nx.draw(T, pos=nx.circular_layout(T))
    '''
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
    #print "T=", T2
    mst_weight = sum([GT[u][v] for u,v in T2])

    T2G= nx.Graph()
    for u in range(len(rep_numbers_cliquewise)):
        for v in range(len(rep_numbers_cliquewise)):
            if (u,v) in T2:
                #print "(u,v)=", u, v
                T2G.add_edge(u,v, weight=1)

    plt.figure(3)
    nx.draw(T2G, pos=nx.circular_layout(T2G))  

    #############################################################################
                   
    L=nx.empty_graph(len(rep_numbers_cliquewise)-1)

    n1=len(Gclique.nodes())

    def mapping(x):
        return x+len(Gclique.nodes())
    L= nx.relabel_nodes(L, mapping)

    for u in range(len(l)):  
        for v in range(len(l)):
            if u in T2G[v]:
                #Gclique.add_edge(l[u],l[v])
                Gclique.add_nodes_from(L.nodes()) 

    #print "Gclique.nodes()",Gclique.nodes()


    ########################### Creating Liaison Graph ##########################
    
    #print l
    l=np.array(l).tolist()
    #print "l = ",l

    for u in range(len(T2G.edges())):
        Gclique.add_edge(l[T2G.edges()[u][0]],n1+u)
        Gclique.add_edge(l[T2G.edges()[u][1]],n1+u)

    Liaisons_num=np.random.randint(1, len(T2G.edges()))
    #print "Liaisons_num =", Liaisons_num

    #iterate= len(L.nodes())-Liaisons_num

    for i in range(len(L.nodes())):
        j= np.random.choice(L.nodes(), 2, replace=False)
        #print j
        #print "j[0] = ", j[0], "j[1] = ", j[1]
        #print Gclique.neighbors(j[0]), Gclique.neighbors(j[1])
        #print "Gclique.degree(j[0])", Gclique.degree(j[0]), "Gclique.degree(j[1])", Gclique.degree(j[1])
        #if Gclique.degree(j[0])+Gclique.degree(j[1])<8 and Gclique.degree(j[0])+Gclique.degree(j[1])>2:
        if Gclique.degree(j[0])+Gclique.degree(j[1])<7 and Gclique.degree(j[0])+Gclique.degree(j[1])>2:
            for n in Gclique.neighbors(j[1]):
                Gclique.add_edge(j[0], n)
                Gclique.add_edge(j[0], n)
            L.remove_node(j[1])
            Gclique.remove_node(j[1])
    #print "L.nodes() = ", L.nodes()
    #print "Gclique.degree(L.nodes())", Gclique.degree(L.nodes())
    #print "Gclique.degree(L.nodes())", sorted(Gclique.degree(L.nodes()).values())

    Gclique=nx.convert_node_labels_to_integers(Gclique,first_label=0)

    #print L.nodes()
    #print Gclique.nodes()
        
    #############################################################################        
      
    #print sorted(nx.degree(Gclique).values())
    degreeList=sorted(nx.degree(Gclique).values())

    counter=collections.Counter(degreeList)
    ''' 
    plt.figure(1)
    
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
 
    #pos=nx.spring_layout(Gclique,iterations=200)
    print "clique_list= ", clique_list
    posx = []; posy = [];
    for i in range(len(clique_list)):
        centerx = np.cos(2*np.pi*i/len(clique_list)) 
        centery = np.sin(2*np.pi*i/len(clique_list))
        x1 = []; y1 = []; 
        for j in range(int(clique_list[i])):
            x1.append(centerx  + 0.2*np.cos(2*np.pi*j/clique_list[i]))
            y1.append(centery  + 0.2*np.sin(2*np.pi*j/clique_list[i]))
        posx.extend(x1); posy.extend(y1);
    
    x1 = []; y1 = []; 
    print "len(L.nodes())=", len(L.nodes())
    for j in range(len(L.nodes())):
        x1.append(0.5*np.cos(2*np.pi*j/len(L.nodes())))
        y1.append(0.5*np.sin(2*np.pi*j/len(L.nodes())))
    posx.extend(x1); posy.extend(y1); 
   
    pos = np.transpose(np.vstack((posx,posy)))
    
    plt.figure(4)
    #nx.draw(Gclique,pos,node_color=colors,node_size=800,cmap=plt.cm.Reds)
    nx.draw(Gclique,pos,node_size=800,cmap=plt.cm.Reds)
    #nx.draw(Gclique,pos=nx.spring_layout(Gclique), node_size=800,cmap=plt.cm.Reds)

    plt.show()
    '''
    A1= nx.to_numpy_matrix(Gclique)
    A = np.array(A1)
    A=np.trunc(A)

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

    with open('StructuralProperties/DegreeDist/L/L'+str(N_init)+'.txt', "a") as text_file:
        text_file.write(str(polycoef[0]) + "\n")
    '''
    print polycoef[0]
    plt.subplot(211)
    plt.plot(xdata,ydata,'.k',xdata,yfit,'-r')
    plt.subplot(212)
    plt.loglog(xdata,ydata,'.k',xdata,yfit,'-r')
    plt.show()        
    
    print "number of nodes: ", math.sqrt(np.size(A))
    '''
    return Gclique
    
#Gclique = Liaison_model3(N_init, sampleN)
 

