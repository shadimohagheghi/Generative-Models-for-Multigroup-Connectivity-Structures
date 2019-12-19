import networkx as nx
from allfunctions import draw_degdist
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from almost_clique import almost_clique 
from numpy import inf
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)
plt.close("all")

##########################################################
##########################################################

def Liaison_model():
    clique_size = [3,4,5,6,7,8,9,10];      # clique sizes
    clique_edges=np.asarray(clique_size)
    clique_edges = np.divide(np.multiply(clique_edges,clique_edges+1),2)
    ep = 0.1;

    clique_num = np.zeros(len(clique_size));  # number of cliques of given size
    for i in range(len(clique_size)):
        clique_num[i] = int(2.5321*(50.0/(clique_size[i])**3));
    clique_num = map(int,clique_num) 
    
    # comment this next line to generate a small sample graph easy to visualize
    #clique_num = [0,7,3,1,0,0,0,0]
 
###########################################################
########################################################### 
 
    def li_num(size):                         # number of liaisons given clique size
        maxli = 2; p = float(size)**2/200;
        a = np.random.binomial(maxli, p, size=1) +1
        return a

###########################################################
###########################################################

    # almost cliques are generated
    Gclique,li,li_cliquewise,clique_sizes,all_liaisons,clique_lead,nstart  = almost_clique(clique_size,clique_num,li_num)

    print(len(clique_lead)), 'cliques formed'
################################################
#### liaison model #############################
    m = 2; ext_li = int(float(sum(clique_num))/m); # deciding number of external liaisons
    Gli = nx.barabasi_albert_graph(ext_li,2);  #only option for external liaison model
    print "Barabasi Albert Graph = ", Gli.edges()
    plt.figure(1)
    nx.draw(Gli, pos=nx.circular_layout(Gli))
    
    limodel_deglist = np.zeros(len(Gli)); 
    for i in range(len(Gli)):
        limodel_deglist[i] = len(Gli[i])
    
    ord_limodel = sorted(range(len(limodel_deglist)),key=lambda x:limodel_deglist[x]) 
    print "ord_limodel = ", ord_limodel
    
    clique_ext_list = np.zeros(sum(clique_num))      
    for i in range(sum(clique_num)): # randomly assign cliques to external liaisons
        clique_ext_list[i] = np.random.randint(ext_li); 
    
    print "clique_ext_list = ", clique_ext_list
    
    cliquenodes = len(Gclique);
    
    for i in range(len(Gli)): 
        for j in range(len(Gli)):
            if j in Gli[i]:
                Gclique.add_edge(cliquenodes+i,cliquenodes+j)
               
    for i in range(ext_li):
        dums = np.where(clique_ext_list==i); 
        for j in range(len(dums[0])):
            for k in range(len(li_cliquewise[dums[0][j]])):
                Gclique.add_edge(cliquenodes+i,li_cliquewise[dums[0][j]][k])
      
      
    degthis,a1,a2 = draw_degdist(Gclique,1,'b',0) 
    
    figcolor = 'b'
     
    plt.figure(2)
     
    plt.scatter((a1),(a2),c=figcolor,marker='o',s=400,alpha=0.5)
    plt.plot((a1),(a2),linewidth=2,c=figcolor) 
    plt.xlabel('node degree',fontsize=30)
    plt.ylabel('number of nodes',fontsize=30)
    plt.axis([-2, 45, -19, 670])

    
    clique_list = [];
    for i in range(len(clique_size)):
        dum =  np.linspace(clique_size[i],clique_size[i], clique_num[i])
        clique_list = np.hstack((clique_list,dum ))
    
    colors = []; c = 0;
    for i in range(len(clique_list)):
        colors.extend(np.linspace(c,c,clique_list[i]))
        c = c + 1
    for i in range(ext_li):
        colors.append(20); c = c + 1
 
    #pos=nx.spring_layout(Gclique,iterations=200)
    posx = []; posy = []; 
    for i in range(len(clique_list)):
        centerx = np.cos(2*np.pi*i/len(clique_list)) 
        centery = np.sin(2*np.pi*i/len(clique_list))
        x1 = []; y1 = []; 
        for j in range(int(clique_list[i])):
            x1.append(centerx  + 0.2*np.cos(2*np.pi*j/clique_list[i]))
            y1.append(centery  + 0.2*np.sin(2*np.pi*j/clique_list[i]))
        posx.extend(x1); posy.extend(y1);
    print ext_li
    x1 = []; y1 = []; 
    print "ext_li=",ext_li
    for j in range(ext_li):
        x1.append(0.5*np.cos(2*np.pi*j/ext_li))
        y1.append(0.5*np.sin(2*np.pi*j/ext_li))
    posx.extend(x1); posy.extend(y1);       
    
    pos = np.transpose(np.vstack((posx,posy)))
   
    plt.figure(3)
    nx.draw(Gclique,pos,node_color=colors,node_size=800,cmap=plt.cm.Blues)
    
    plt.show() 
    print 'diameter of liaison network is', nx.diameter(Gclique)
    print 'avg clustering coeff is', nx.average_clustering(Gclique)
    print 'avg shortest path length', nx.average_shortest_path_length(Gclique)
    plt.show()
    
    return Gclique

     
        
Gclique = Liaison_model()
    
    
    
    
    
    
    
    
    
    
    
    