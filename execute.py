# Parameters
from Classes import *
from graph import *
import time
random.seed(0)
import copy
num_nodes = 10
nearest_neighbors = 2
rewiring_probability = 0.2
weight = 0.75
ws_graph = generate_constant_weighted_watts_strogatz_graph(num_nodes, nearest_neighbors, rewiring_probability, weight)

iter=10

layers=7
diff_prob=0.6
tree=Tree(ws_graph)
num_nodes=len(ws_graph.nodes)
beta=0.6
gamma=14
affected=[]


ind=np.random.choice([0,1,2],num_nodes,p=[1,0,0])
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
            util_algo=0

            util_greedy=0
            util_greedy_1=0

            affected_1 = copy.deepcopy(affected)
            total_aff_1 = copy.deepcopy(total_aff)
            labels_1 = copy.deepcopy(labels)
            flags_1 = copy.deepcopy(flags)

            affected_2 = copy.deepcopy(affected)
            total_aff_2 = copy.deepcopy(total_aff)
            labels_2 = copy.deepcopy(labels)
            flags_2 = copy.deepcopy(flags)

            continue
        else:
            util_1.append(util_max)

        empty_indices=[i for i in range(len(affected)) if (len(affected[i])==0)]
        labels= [labels[i] for i in range(len(labels)) if i not in empty_indices]
        total_aff= [total_aff[i] for i in range(len(total_aff)) if i not in empty_indices]
        affected= [affected[i] for i in range(len(affected)) if i not in empty_indices]
        flags= np.delete(flags,empty_indices,0)

        empty_indices_1=[i for i in range(len(affected_1)) if (len(affected_1[i])==0)]
        labels_1= [labels_1[i] for i in range(len(labels_1)) if i not in empty_indices_1]
        total_aff_1= [total_aff_1[i] for i in range(len(total_aff_1)) if i not in empty_indices_1]
        affected_1= [affected_1[i] for i in range(len(affected_1)) if i not in empty_indices_1]
        flags_1= np.delete(flags_1,empty_indices_1,0)

        empty_indices_2=[i for i in range(len(affected_2)) if (len(affected_2[i])==0)]
        labels_2= [labels_2[i] for i in range(len(labels_2)) if i not in empty_indices_2]
        total_aff_2= [total_aff_2[i] for i in range(len(total_aff_2)) if i not in empty_indices_2]
        affected_2= [affected_2[i] for i in range(len(affected_2)) if i not in empty_indices_2]
        flags_2= np.delete(flags_2,empty_indices_2,0)

        
        
        # Sending to Oracle!
        news=News(tree,affected,rel,labels,flags,beta,gamma,total_aff)
        indices= Algorithm(news).select_news()


        # Sending to Oracle_Greedy!
        news_1=News(tree,affected_1,rel,labels_1,flags_1,beta,gamma,total_aff_1)
        indices_1=Algorithm_min(news_1).select_news()


        news_2=News(tree,affected_2,rel,labels_2,flags_2,beta,gamma,total_aff_2)
        indices_2= Greedy(news_2).select_news()

        # print(t,labels,affected,indices,indices_1)

        util_algo+=np.sum((1-np.array(labels)[indices])*np.array([tree.val_calc(affected[i],total_aff[i]) for i in indices])) - gamma*len(indices)
        labels= [labels[i] for i in range(len(labels)) if i not in indices]  # list
        affected= [affected[i] for i in range(len(affected)) if i not in indices] # list of lists
        total_aff= [total_aff[i] for i in range(len(total_aff)) if i not in indices]
        flags= np.delete(flags,indices,0) # array
        util_2.append(util_algo)



        util_greedy+=np.sum((1-np.array(labels_1)[indices_1])*np.array([tree.val_calc(affected_1[i],total_aff_1[i]) for i in indices_1])) - gamma*len(indices_1)
        labels_1= [labels_1[i] for i in range(len(labels_1)) if i not in indices_1]  # list
        affected_1= [affected_1[i] for i in range(len(affected_1)) if i not in indices_1] # list of lists
        total_aff_1= [total_aff_1[i] for i in range(len(total_aff_1)) if i not in indices_1]
        flags_1= np.delete(flags_1,indices_1,0) # array
        util_3.append(util_greedy)



        util_greedy_1+=np.sum((1-np.array(labels_2)[indices_2])*np.array([tree.val_calc(affected_2[i],total_aff_2[i]) for i in indices_2])) - gamma*len(indices_2)
        labels_2= [labels_2[i] for i in range(len(labels_2)) if i not in indices_2]  # list
        affected_2= [affected_2[i] for i in range(len(affected_2)) if i not in indices_2] # list of lists
        total_aff_2= [total_aff_2[i] for i in range(len(total_aff_2)) if i not in indices_2]
        flags_2= np.delete(flags_2,indices_2,0) # array
        util_4.append(util_greedy_1)


        print(f'Algo {util_algo}, {np.array(labels)[indices]}')
        print(f'Algo_approx {util_greedy}, {np.array(labels_1)[indices_1]}')
        print(f'Greedy_Algo {util_greedy_1}, {np.array(labels_2)[indices_2]}')
        # News infection


        news=News(tree,affected,rel,labels,flags,beta,gamma,total_aff)
        affected,flags=news.infected()
        iterate=0
        for ind in affected:
            for i in ind:
                total_aff[iterate][i]=True
            iterate+=1


        news_1=News(tree,affected_1,rel,labels_1,flags_1,beta,gamma,total_aff_1)
        affected_1,flags_1=news_1.infected()
        iterate_1=0
        for ind_1 in affected_1:
            for i in ind_1:
                total_aff_1[iterate_1][i]=True
            iterate_1+=1

        news_2=News(tree,affected_2,rel,labels_2,flags_2,beta,gamma,total_aff_2)
        affected_2,flags_2=news_2.infected()
        iterate_2=0
        for ind_2 in affected_2:
            for i in ind_2:
                total_aff_2[iterate_2][i]=True
            iterate_2+=1
        
        if (t<9 and t>0):
            # new=np.random.randint(0,2**(layers//1.5),2)
            # new_labels=np.random.choice([True,False],2,p=[1-beta,beta])
            new=np.random.randint(0,num_nodes,1)
            new_labels=np.random.choice([True,False],1,p=[1-beta,beta])

            flags=np.vstack((flags,-1*np.ones((1,num_nodes))))
            labels=np.concatenate((labels,new_labels))
            affected.extend([[item] for item in new])
            total_aff.append([True if item==new[0] else False for item in range(num_nodes)])

            flags_1=np.vstack((flags_1,-1*np.ones((1,num_nodes))))
            labels_1=np.concatenate((labels_1,new_labels))
            affected_1.extend([[item] for item in new])
            total_aff_1.append([True if item==new[0] else False for item in range(num_nodes)])

            flags_2=np.vstack((flags_2,-1*np.ones((1,num_nodes))))
            labels_2=np.concatenate((labels_2,new_labels))
            affected_2.extend([[item] for item in new])
            total_aff_2.append([True if item==new[0] else False for item in range(num_nodes)])

            util_max+= max(np.sum((1-np.array(new_labels))*np.array([tree.val_calc([i],[True if item==i else False for item in range(num_nodes)]) for i in new]))-len(np.argwhere(new_labels==0).reshape(-1))*gamma,0)
    if(repeat==0):
        u_1=np.array(util_1)
        u_2=np.array(util_2)
        u_3=np.array(util_3)
        u_4=np.array(util_4)
    else:
        u_1+=np.array(util_1)
        u_2+=np.array(util_2)
        u_3+=np.array(util_3)
        u_4+=np.array(util_4)
    print(repeat)

u_1=np.array(u_1)/iter
u_2=np.array(u_2)/iter
u_3=np.array(u_3)/iter
u_4=np.array(u_4)/iter

np.savez('results_14.npz',u_1=u_1, u_2=u_2, u_3=u_3, u_4=u_4)
# plt.plot(u_1, label='Algo_Opt')
plt.plot(u_2, label='Algo')
plt.plot(u_3, label='Algo_approx')
plt.plot(u_4, label='Greedy')

plt.xlabel('Time Step')
plt.ylabel('Avg Utility (100 iterations)')
plt.title('Gamma = 14 (Reliability = 75%)')
plt.legend()
plt.savefig('Trial.png')
plt.show()