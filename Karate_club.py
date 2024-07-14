from Classes import *
from graph import *
random.seed(1)


# group_order = 20
# group_generators = [2]

# ws_graph = generate_cayley_graph("Z15", group_generators, group_order, 0.9)
num_nodes = 10
nearest_neighbors = 2
rewiring_probability = 0.2
weight = 0.7
# ws_graph = generate_constant_weighted_watts_strogatz_graph(num_nodes, nearest_neighbors, rewiring_probability, weight)
ws_graph = generate_constant_weighted_karate_club_graph(weight)
iter=5
pos = nx.spring_layout(ws_graph)  # You can use other layouts as well
nx.draw(ws_graph, pos, with_labels=True, node_color='lightblue', node_size=800)

# Extract edge weights
edge_labels = {(u, v): ws_graph[u][v]['wt'] for u, v in ws_graph.edges()}

# Draw edge labels
nx.draw_networkx_edge_labels(ws_graph, pos, edge_labels=edge_labels)

plt.title("Karate Club Graph")
plt.savefig('Karate_Club_Graph.png')
plt.close()

layers=7
diff_prob=0.6
tree=Tree(ws_graph)
num_nodes=len(ws_graph.nodes)
beta=0.6
gamma=10
affected=[]


ind=np.random.choice([0,1,2],num_nodes,p=[0,0,1])
t=np.array([[0.75,0.75],[0.9,0.1],[0.9999,0.9999]])
rel=t[ind]

u_1=[]
u_2=[]
u_3=[]
u_4=[]
for repeat in range(iter):
    util_1=[0]
    util_2=[0]
    util_3=[0]
    util_4=[0]
    for t in range(15):
        if (t==0):
            total_aff=[[False]*num_nodes]
            affected=[[np.random.randint(num_nodes)]]
            total_aff[0][affected[0][0]]=True
            labels=np.random.choice([True,False],1,p=[1-beta,beta])
            flags=-1*np.ones((1,num_nodes))
            util_max=max(np.sum((1-np.array(labels))*np.array([tree.val_calc(i,total_aff[0]) for i in affected]))-len(np.argwhere(labels==0).reshape(-1))*gamma,0)
            print(util_max)
            # util_algo=0

            util_greedy=0
            util_greedy_1=0

            # list_algo=[0] #Gives active News for Algo algorithm
            list_greedy=[0]
            list_approx=[0]

            continue
        else:
            util_1.append(util_max)


        empty_indices=[i for i in range(len(affected)) if (len(affected[i])==0)]
        labels= [labels[i] for i in range(len(labels)) if i not in empty_indices]
        total_aff= [total_aff[i] for i in range(len(total_aff)) if i not in empty_indices]
        affected= [affected[i] for i in range(len(affected)) if i not in empty_indices]
        flags= np.delete(flags,empty_indices,0)

        if len(empty_indices)>0:
            # list_algo=update(list_algo, empty_indices)
            list_approx=update(list_approx, empty_indices)
            list_greedy=update(list_greedy, empty_indices)


        # affected_algo, labels_algo, total_aff_algo, flags_algo = [affected[i] for i in list_algo], [labels[i] for i in list_algo], [total_aff[i] for i in list_algo], flags[list_algo]
        affected_approx, labels_approx, total_aff_approx, flags_approx = [affected[i] for i in list_approx], [labels[i] for i in list_approx], [total_aff[i] for i in list_approx], flags[list_approx]
        affected_greedy, labels_greedy, total_aff_greedy, flags_greedy = [affected[i] for i in list_greedy], [labels[i] for i in list_greedy], [total_aff[i] for i in list_greedy], flags[list_greedy]



        # Sending to Oracle!
        # print('start1')
        # news=News(tree,affected_algo,rel,labels_algo,flags_algo,beta,gamma,total_aff_algo)
        # indices=Algorithm(news).select_news()
        # algo_selected = [list_algo[i] for i in indices]
        # print('start2')
        # Sending to Oracle_Greedy!
        news_1=News(tree,affected_approx,rel,labels_approx,flags_approx,beta,gamma,total_aff_approx)
        indices_1=New(news_1).select_news()
        approx_selected = [list_approx[i] for i in indices_1]
        print('start3')

        news_2=News(tree,affected_greedy,rel,labels_greedy,flags_greedy,beta,gamma,total_aff_greedy)
        indices_2=Greedy(news_2).select_news()
        greedy_selected=[list_greedy[i] for i in indices_2]


        # list_algo=[i for i in list_algo if i not in algo_selected]
        list_approx=[i for i in list_approx if i not in approx_selected]
        list_greedy=[i for i in list_greedy if i not in greedy_selected]

        # util_algo+=np.sum((1-np.array(labels)[algo_selected])*np.array([tree.val_calc(affected[i],total_aff[i]) for i in algo_selected])) - gamma*len(algo_selected)
        util_greedy+=np.sum((1-np.array(labels)[approx_selected])*np.array([tree.val_calc(affected[i],total_aff[i]) for i in approx_selected])) - gamma*len(approx_selected)
        util_greedy_1+=np.sum((1-np.array(labels)[greedy_selected])*np.array([tree.val_calc(affected[i],total_aff[i]) for i in greedy_selected])) - gamma*len(greedy_selected)

        print(f'List_Greedy: {list_greedy}, List_approx: {list_approx}')
        print(f'Greedy:{greedy_selected}, Approx: {approx_selected}, label:{labels}' )
        print(f'Util_Greedy: {util_greedy_1}, Util_Approx: {util_greedy}')

        # util_2.append(util_algo)
        util_3.append(util_greedy)
        util_4.append(util_greedy_1)

        common= [i for i in indices_1 if (i in indices_2)]
        if len(common)>0:
            labels= [labels[i] for i in range(len(labels)) if i not in common]
            total_aff= [total_aff[i] for i in range(len(total_aff)) if i not in common]
            affected= [affected[i] for i in range(len(affected)) if i not in common]
            flags= np.delete(flags,common,0)
        
        # list_algo=update(list_algo, common)
        list_approx=update(list_approx, common)
        list_greedy=update(list_greedy, common)


        news=News(tree,affected,rel,labels,flags,beta,gamma,total_aff)
        affected,flags=news.infected()
        iterate=0
        for ind in affected:
            for i in ind:
                total_aff[iterate][i]=True
            iterate+=1

        if (t<9 and t>0):

            new=np.random.randint(0,num_nodes,1)
            new_labels=np.random.choice([True,False],1,p=[0.4,0.6])

            flags=np.vstack((flags,-1*np.ones((1,num_nodes))))
            labels=np.concatenate((labels,new_labels))
            affected.extend([[item] for item in new])
            total_aff.append([True if item==new[0] else False for item in range(num_nodes)])

            # list_algo.append(len(labels)-1)
            list_approx.append(len(labels)-1)
            list_greedy.append(len(labels)-1)

            print(f'Node: {new[0]}, Val: {tree.val_calc(affected[-1],total_aff[-1])}')

            util_max+= max(np.sum((1-np.array(new_labels))*np.array([tree.val_calc([i],[True if item==i else False for item in range(num_nodes)]) for i in new]))-len(np.argwhere(new_labels==0).reshape(-1))*gamma,0)
    if(repeat==0):
        u_1=np.array(util_1)
        # u_2=np.array(util_2)
        u_3=np.array(util_3)
        u_4=np.array(util_4)
    else:
        u_1+=np.array(util_1)
        # u_2+=np.array(util_2)
        u_3+=np.array(util_3)
        u_4+=np.array(util_4)
    print(repeat)

u_1=np.array(u_1)/iter
# u_2=np.array(u_2)/iter
u_3=np.array(u_3)/iter
u_4=np.array(u_4)/iter

np.savez('Karate.npz', u_1=u_1, u_3=u_3, u_4=u_4)
plt.plot(u_1, label = 'Opt')
# plt.plot(u_2, label= 'Algo')
plt.plot(u_3, label= 'Algo_approx')
plt.plot(u_4, label= 'Greedy')

plt.xlabel('Time Step')
plt.ylabel('Avg Utility (100 iterations)')
plt.title('Karate Club Graph: IP = 70% & Gamma = 5.35 (Reliability = 100%)')
plt.legend()
plt.savefig('Karate.png')
plt.close()