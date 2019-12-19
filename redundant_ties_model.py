import networkx as nx
from allfunctions import draw_degdist
import numpy as np
import math
import collections
import matplotlib 
import matplotlib.pyplot as plt
from almost_clique import almost_clique
from numpy import inf
from random import randint
from random import random
from MinimumSpanningTree import MinimumSpanningTree as mst
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
plt.close("all")

##########################################################

def redundant_ties_model():
    clique_size = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];      # clique sizes    
    clique_edges=np.asarray(clique_size)
    clique_edges = np.divide(np.multiply(clique_edges,clique_edges+1),2)
    ep = 0.1; 
    
    clique_num = np.zeros(len(clique_size));  # number of cliques of given size
    for i in range(len(clique_size)):
        clique_num[i] = int(2.5321*(40.0/(clique_size[i])**3));
    clique_num = map(int,clique_num) 
    
    # comment this next line to generate a small sample graph easy to visualize
    #clique_num = [0,7,3,1,0,0,0,0]
    
    ##########################################################

    def r_num(size):
        '''                         # number of representetives given clique size
        maxrep = size-1; p = (size**2)/100.0;
        #a = np.random.binomial(3, size/10.0) + 1
        print p
        if size == 3:
            a = np.random.binomial(1, p) + 1  
        elif size >3 and size <6:
            a = np.random.binomial(2, p) + 1
        elif p>=1:
            a = np.random.binomial(3, 0.95) + 1
        else:
            a = np.random.binomial(3, p) + 1
        '''
        if size == 3:
            a =1  
        elif size >3 and size <6:
            a = np.random.randint(1, 2)
        elif size >5 and size <11:
            a = np.random.randint(2, 3)
        else:
            a = np.random.randint(3, 4)
        return a

    ###########################################################

    # almost cliques are generated
    Gclique,r,r_cliquewise,clique_sizes,rep_numbers_cliquewise, reps_ordered_cliquewise, clique_start = almost_clique(clique_size,clique_num,r_num)
    
    print(len(rep_numbers_cliquewise)), 'cliques formed'
    
    clique_list = []
    for i in range(len(clique_size)):
        dum =  np.linspace(clique_size[i],clique_size[i], clique_num[i])
        clique_list = np.hstack((clique_list,dum ))
    cn = len(clique_list)
  
    ########################### Spanning Tree with Power Law #####################
    '''
    global T
    T=nx.random_powerlaw_tree(cn, tries=1000000)
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
                T2G.add_edge(u,v)

    plt.figure(1)
    nx.draw(T2G, pos=nx.circular_layout(T2G))  

    #############################################################################
    CBnodes1=[]
    CBnodes2=[]
    updated_CB=[]
    updated_CB_edges=[]
    for u in range(0, cn): 
        for v in T2G[u]:
            CBnodes1=[]
            CBupdated1=[]
            CBnodes2=[]
            updated_CB_edges=[]
            CB_u=[]
            if v>=u:
                CB=nx.complete_bipartite_graph(rep_numbers_cliquewise[u], rep_numbers_cliquewise[v])
                print 'CB', CB.edges()
                for i in range(rep_numbers_cliquewise[u]):
                    CBnodes1.append(CB.nodes()[i])

                CBnodes1=np.array(CBnodes1)

                for i in range(rep_numbers_cliquewise[u]):
                    CBupdated1= r_cliquewise[u]   
                #print "CBnodes1", CBnodes1
                #print "CBupdated1", CBupdated1

                for i in range(rep_numbers_cliquewise[v]):
                    CBnodes2.append(CB.nodes()[i])

                CBnodes2=np.array(CBnodes2)

                for i in range(rep_numbers_cliquewise[v]):
                    CBupdated2= r_cliquewise[v] 

                #print "CBnodes2", CBnodes2
                #print "CBupdated2", CBupdated2

                updated_CB= np.union1d(CBupdated1, CBupdated2)
                print "updated_CB", updated_CB

                for i in range(rep_numbers_cliquewise[u]):
                    for j in range(rep_numbers_cliquewise[v]):
                        updated_CB_edges.append([updated_CB[i], updated_CB[rep_numbers_cliquewise[u]+j]])

                print "updated_CB_edges", updated_CB_edges

                edges=randint(1,len(CB.edges()))
                print "edges", edges
                e=0 
                e_index= np.random.choice(len(CB.edges()), edges, replace=False)
                print "e_index", e_index
                '''
                for i in range(rep_numbers_cliquewise[u]):  
                    for j in range(rep_numbers_cliquewise[v]):
                        if e<edges:
                            n1= reps_ordered_cliquewise[u][0:i]
                            n2= reps_ordered_cliquewise[v][0:j]
                            Gclique.add_edge(clique_start[u]+reps_ordered_cliquewise[u][i], clique_start[v]+reps_ordered_cliquewise[v][j])
                            e=e+1
                '''
                print len(updated_CB_edges)
                for i in range(len(e_index)):  
                    if e<edges:
                        print updated_CB_edges[e_index[i]]
                        CB_u.append(updated_CB_edges[e_index[i]])
                        Gclique.add_edges_from(CB_u)
                        e=e+1
    #degthis,a1,a2 = draw_degdist(Gclique,2,'b',0) 
    '''   
    figcolor = 'purple'
        
    plt.figure(3)
    
        
    plt.scatter((a1),(a2),c=figcolor,marker='o',s=400,alpha=0.5)
    plt.plot((a1),(a2),linewidth=2,c=figcolor) 
    plt.xlabel('node degree',fontsize=30)
    plt.ylabel('number of nodes',fontsize=30)
    plt.axis([0, 4, -0.1, 7])
    '''
    colors = []; c = 0;
    for i in range(len(clique_list)):
        colors.extend(np.linspace(c,c,clique_list[i]))
        c = c + 1
    
    #pos=nx.spring_layout(Gclique,iterations=200)
    posx = []; posy = [];
    for i in range(len(clique_list)):     
        centerx = np.cos(2*np.pi*i/len(clique_list)) 
        centery = np.sin(2*np.pi*i/len(clique_list))
        x1 = []; y1 = []; 
        for j in range(int(clique_list[i])):
            x1.append(centerx  + 0.2*np.cos(2*np.pi*j/clique_list[i]))       
            y1.append(centery  + 0.2*np.sin(2*np.pi*j/clique_list[i]))
        posx.extend(x1); posy.extend(y1)
        
        #posx.extend(centerx+np.random.rand(clique_list[i])*0.3)
        #posy.extend(centery+np.random.rand(clique_list[i])*0.3)
    
    pos = np.transpose(np.vstack((posx,posy)))
    
    plt.figure(2)
    nx.draw(Gclique,pos,node_color=colors,node_size=800,cmap=plt.cm.Purples)
    
    plt.show()  
     
    print 'diameter of redundant ties network is', nx.diameter(Gclique)
    print 'avg clustering coeff is', nx.average_clustering(Gclique)
    print 'avg shortest path length', nx.average_shortest_path_length(Gclique)
        
    return Gclique             
    
Gclique = redundant_ties_model()            
            

