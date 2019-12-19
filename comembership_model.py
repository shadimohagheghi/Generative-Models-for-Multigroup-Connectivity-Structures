import networkx as nx
from allfunctions import draw_degdist
import numpy as np
import matplotlib.pyplot as plt
import math
import collections
import matplotlib
import random
from random import random
from MinimumSpanningTree import MinimumSpanningTree as mst
from almost_clique import almost_clique
from numpy import inf
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
plt.close("all")

##########################################################
 
def comembership_model():
    clique_size = [3,4,5,6,7,8,9,10]     # clique sizes
            
    clique_edges=np.asarray(clique_size)
    clique_edges = np.divide(np.multiply(clique_edges,clique_edges+1),2)
    ep = 0.1
    
    clique_num = np.zeros(len(clique_size)) # number of cliques of given size
    for i in range(len(clique_size)):
        clique_num[i] = int(2.5321*(70.0/(clique_size[i])**3))
    clique_num = map(int,clique_num) 
    # comment this next line to generate a small sample graph easy to visualize
    #clique_num = [0,7,3,1,0,0,0,0]
    
##########################################################

    def c_num(size):                         # number of comembers given clique size
        if size == 3:
            a =1  
        elif size >3 and size <6:
            a = np.random.randint(2, 3)
        elif size >5 and size <11:
            a = np.random.randint(3, 4)
        else:
            a = 4
        return a

###########################################################

# almost cliques are generated
    Gclique,r,r_cliquewise,clique_sizes,rep_numbers_cliquewise, reps_ordered_cliquewise, clique_start = almost_clique(clique_size,clique_num,c_num)
    print(len(rep_numbers_cliquewise)), 'cliques formed'
    
    ####### MST of Randomly Weighted Undirected Complete Graph on n nodes ########

    GT = {}
    for u in range(len(rep_numbers_cliquewise)):
        GT[u] = {}
    for u in range(len(rep_numbers_cliquewise)):
        for v in range(u):
            rand = random()
            GT[u][v] = rand
            GT[v][u] = rand
    T2 = mst(GT)
    mst_weight = sum([GT[u][v] for u,v in T2])

    T2G= nx.Graph()
    for u in range(len(rep_numbers_cliquewise)):
        for v in range(len(rep_numbers_cliquewise)):
            if (u,v) in T2:
                print "(u,v)=", u, v
                T2G.add_edge(u,v)

    plt.figure(2)
    nx.draw(T2G, pos=nx.circular_layout(T2G))  

    #############################################################################  
    cn=len(rep_numbers_cliquewise)  
    owner = np.linspace(len(Gclique)+1,len(Gclique)+1,len(Gclique)) 
    done = np.zeros((cn,cn))

    roots=[]
    print "r",r
    print "len(r)", len(r)
    #roots_no=randint(1,len(CB.edges()))

    roots=np.random.choice(5, 3, replace=False)
    print "roots", roots

    for i in range(cn):
        neb = T2G.neighbors(i)
        print 'Neighbors of ', i, '=', neb
        for j in range(len(neb)):
            if done[i,neb[j]] == 0 and done[neb[j],i] == 0:
                done[i,neb[j]] = 1; done[neb[j],i] = 0

                a1= rep_numbers_cliquewise[i]
                a2= rep_numbers_cliquewise[T2G.neighbors(i)[j]]
                #print 'a1= ', a1 
                #print 'a2= ', a2 

                a = min(a1,a2)
                print 'a= ', a 
                
                nl1 = reps_ordered_cliquewise[i][0:a]; nl2 = reps_ordered_cliquewise[neb[j]][0:a]

                print 'nl1 = ', nl1
                print 'nl2 = ', nl2
                
                for k in range(len(nl1)):
                    Gclique.add_edge(clique_start[i]+nl1[k],clique_start[neb[j]]+nl2[k])
                    owner[clique_start[i]+nl1[k]] = clique_start[neb[j]]+nl2[k]
                    print "clique_start[i]+nl1[k]", clique_start[i]+nl1[k]
                    print "clique_start[neb[j]]+nl2[k]", clique_start[neb[j]]+nl2[k]

    print "owner = ", owner  

    c = 0  
    clique_mem = np.zeros(len(clique_sizes))
    for i in range(len(clique_sizes)):
        for k in range(int(clique_sizes[i])):
            if owner[c] ==  len(Gclique)+1:
                print "c", c
                clique_mem[i] = clique_mem[i] + 1
            c=c+1
                    
    print "clique_mem = ", clique_mem
    print "n = ", sum(clique_mem)
    
    for j in range(len(Gclique)):
        if owner[j] != len(Gclique)+1:
            for k in range(len(Gclique.neighbors(j))):
                Gclique.add_edge(Gclique.neighbors(j)[k],owner[j])             
    keep = np.where(owner==len(Gclique)+1); keep = list(keep);
    print "keep", keep
    Gclique = Gclique.subgraph(keep[0])

    print "owner = ", owner 
    
    degthis,a1,a2 = draw_degdist(Gclique,1,'b',0) 
         
    figcolor = 'g'
    
    plt.figure(3)
     
    plt.scatter((a1),(a2),c=figcolor,marker='o',s=400,alpha=0.5)
    plt.plot((a1),(a2),linewidth=2,c=figcolor) 
    plt.xlabel('node degree',fontsize=30)
    plt.ylabel('number of nodes',fontsize=30)
    plt.axis([-19, 200, -19, 350])
    
    colors = []; c = 0;
    for i in range(len(clique_mem)):
        colors.extend(np.linspace(c,c,clique_mem[i]))
        c = c + 1
 
    clique_list1 = [];
    for i in range(len(clique_size)):
        dum =  np.linspace(clique_size[i],clique_size[i], clique_num[i])
        clique_list1 = np.hstack((clique_list1,dum ))
    #pos=nx.spring_layout(Gclique,iterations=200)
    posx = []; posy = [];
    for i in range(len(clique_list1)): 
        centerx = np.cos(2*np.pi*i/len(clique_list1)) 
        centery = np.sin(2*np.pi*i/len(clique_list1))
        x1 = []; y1 = []; 
        for j in range(int(clique_list1[i])):
            x1.append(centerx  + 0.2*np.cos(2*np.pi*j/clique_list1[i]))
            y1.append(centery  + 0.2*np.sin(2*np.pi*j/clique_list1[i]))
        posx.extend(x1); posy.extend(y1);
   
    pos = np.transpose(np.vstack((posx,posy)))
   
    plt.figure(4)
    nx.draw(Gclique,pos,node_color = colors,node_size=800,cmap=plt.cm.Greens)
    
    plt.show()
    print 'diameter of comembership network is', nx.diameter(Gclique)
    print 'avg clustering coeff is', nx.average_clustering(Gclique)
    print 'avg shortest path length', nx.average_shortest_path_length(Gclique)
    
    return Gclique
    
Gclique = comembership_model()            
            
            