import math
import multiprocessing
import numpy as np
import networkx as nx
from copy import deepcopy


def to_base_n(x, n, num):
    """
    Converts a number x to base n
    :param x: number to convert
    :param n: base to convert to
    :return: list of integers (0 to n-1) representing x in base n
    """
    if x == 0:
        return np.array([0]*num)

    symbols = []
    while x > 0:
        symbols.append(x % n)
        x //= n
    symbols += (num-len(symbols))*[0]
    return np.array(symbols[::-1])


def update(lst, indices):
    lst_new = [i for i in lst if i not in indices]
    num = np.zeros(len(lst_new), int)
    min = 0
    search = 0
    while (min < len(lst_new) and search < len(indices)):
        if (lst_new[min] < indices[search]):
            min += 1
            continue
        else:
            num[min:] += 1
            search += 1

    return list(np.array(lst_new)-num)


# inf=> N= [0,1] => [2,3,4], [0,0,0.5...]
def monte_carlo_expected_spread(graph, inf, total_aff, num_simulations=1000):
    spread = 0
    total = deepcopy(total_aff)
    if (len(inf) == 0):
        return 0
    for _ in range(num_simulations):
        total = deepcopy(total_aff)
        tree = Tree(graph)
        infected = tree.affectednodes(inf, total_aff)[0]
        spread += len(infected)
        # print(infected)
        while (len(infected) > 0):
            for i in infected:
                total_aff[i] = True
            infected = tree.affectednodes(infected, total_aff)[0]
            # print(infected)
            spread += len(infected)
    return spread / num_simulations


def affectednodes(graph, N, total_aff):  # N= [0,1] => [2,3,4], [0,0,0.5...]
    # Gives samples of infected nodes indices
    # also gives array (all_nodes_length) of probabilities of influence
    P = nx.linalg.graphmatrix.adjacency_matrix(
        graph, nodelist=list(graph.nodes), weight='wt').toarray()
    prob_matrix = P[:, N]  # pij= prob of jth column infecting i
    probabilities = []
    for i in prob_matrix:
        probabilities.append(1-np.prod(1-i))
    probabilities = np.array(probabilities)
    random = np.random.random(P.shape[0])
    # G_inf=np.array([graph.nodes[i]['inf'] for i in graph.nodes]) #inf
    affected = np.argwhere((probabilities > random) & (
        np.array(total_aff) == False)).reshape(-1)

    return affected, probabilities


def worker_function(data_in):
    total_aff, g, inf, spread = data_in
    total = deepcopy(total_aff)

    for i in inf:  # this is imp for fut_val_calc, one-step-look ahead
        total[i] = True

    tree = Tree(g)
    infected = tree.affectednodes(inf, total)[0]
    spread += len(infected)
    # print(infected)
    while (len(infected) > 0):
        for i in infected:
            total[i] = True
        infected = tree.affectednodes(infected, total)[0]
        # print(infected)
        spread += len(infected)
    return spread


def parallelize_task(tasks):

    with multiprocessing.Pool() as pool:
        # Distribute the tasks across processes and collect the results
        results = pool.map(worker_function, tasks)
    return np.mean(list(results))


# inf=> N= [0,1] => [2,3,4], [0,0,0.5...]
def monte_carlo_expected_spread(g, inf, total_aff, num_simulations=1000):
    spread = 0
    total = deepcopy(total_aff)
    if (len(inf) == 0):
        return 0

    tasks = [(total_aff, g, inf, spread) for i in range(num_simulations)]
    return parallelize_task(tasks)


# inf=> N= [0,1] => [2,3,4], [0,0,0.5...]
def monte_carlo_OG(g, inf, total_aff, num_simulations=1000):
    spread = 0
    total = deepcopy(total_aff)
    if (len(inf) == 0):
        return 0

    for _ in range(num_simulations):
        total = deepcopy(total_aff)

        for i in inf:  # this is imp for fut_val_calc, one-step-look ahead
            total[i] = True
        tree = Tree(g)
        infected = tree.affectednodes(inf, total)[0]
        spread += len(infected)

        while (len(infected) > 0):
            for i in infected:
                total[i] = True
            infected = tree.affectednodes(infected, total)[0]
            # print(infected)
            spread += len(infected)
    return spread


class Tree():
    def __init__(self, graph):
        self.graph = graph
        self.P = nx.linalg.graphmatrix.adjacency_matrix(
            self.graph, nodelist=list(self.graph.nodes), weight='wt').toarray()

    def affectednodes(self, N, total_aff):  # N= [0,1] => [2,3,4], [0,0,0.5...]
        # Gives samples of infected nodes indices
        # also gives array (all_nodes_length) of probabilities of influence

        prob_matrix = self.P[:, N]  # pij= prob of jth column infecting i
        probabilities = []
        for i in prob_matrix:
            probabilities.append(1-np.prod(1-i))
        probabilities = np.array(probabilities)
        random = np.random.random(self.P.shape[0])
        # G_inf=np.array([self.graph.nodes[i]['inf'] for i in self.graph.nodes]) #inf
        affected = np.argwhere((probabilities > random) & (
            np.array(total_aff) == False)).reshape(-1)

        return affected, probabilities

    def val_next(self, N, total_aff):  # N= [0,1] => [2,3,4], [0,0,0.5...]
        # Gives samples of infected nodes indices
        # also gives array (all_nodes_length) of probabilities of influence
        val_now = self.val_calc(inf=N, total_aff=deepcopy(total_aff))
        prob_matrix = self.P[:, N]  # pij= prob of jth column infecting i
        probabilities = []
        for i in prob_matrix:
            probabilities.append(1-np.prod(1-i))
        probabilities = np.array(probabilities)
        affected = np.argwhere((probabilities > 1e-3) &
                               (np.array(total_aff) == False)).reshape(-1)
        val_tplus = val_now - np.sum(probabilities[affected])
        return max(val_tplus, 0)

    def Estimator(self, P_t, prob_inf, rel):
        num_inf = len(prob_inf)
        est_f = 0
        est_t = 0
        for iter in range(3**num_inf):
            array = to_base_n(iter, 3, num_inf)-1
            prob_f = 1
            prob_t = 1
            unconditional_est = 1
            for i in range(len(array)):
                if array[i] == -1:
                    prob_f *= (1-prob_inf[i])
                    prob_t *= (1-prob_inf[i])
                elif array[i] == 0:
                    prob_f *= (prob_inf[i])*(1-rel[i, 0])
                    prob_t *= (prob_inf[i])*(rel[i, 1])
                    unconditional_est *= (rel[i, 1])/(1-rel[i, 0])
                else:
                    prob_f *= (prob_inf[i])*(rel[i, 0])
                    prob_t *= (prob_inf[i])*(1-rel[i, 1])
                    unconditional_est *= (1-rel[i, 1])/(rel[i, 0])
            unconditional_est = 1/(1 + ((1/P_t)-1)*unconditional_est)
            # print(f'array: {array}, Probabaility if False News: {prob_f} , Probability if True News: {prob_t}, Estimate: {unconditional_est}')
            est_f += prob_f*unconditional_est
            est_t += prob_t*unconditional_est
        return est_f, est_t

    def fut_val_calc(self, N, total_aff):
        # Need to check current nodes to be added or not
        self.lst = []
        self.recur(N, [], 0, len(N), total_aff)
        # [self.lst.insert(pos, elem) for pos, elem in zip(range(1,2*len(self.lst),3), self.lst[0::2])]
        return self.lst

    def recur(self, N, indices, i, maxim, total_aff):
        if (i != maxim):
            ind = deepcopy(indices)
            ind.append(N[i])
            len_lst_1 = len(self.lst)
            self.recur(N, ind, i+1, maxim, total_aff)
            len_lst_2 = len(self.lst)
            self.lst += self.lst[len_lst_1:len_lst_2]
            self.recur(N, indices, i+1, maxim, total_aff)

        else:
            self.lst.append(self.val_calc(indices, total_aff))
            return

    # inf=> N= [0,1] => [2,3,4], [0,0,0.5...]
    def val_calc(self, inf, total_aff, num_simulations=2000, add_current=False):
        # Summed up only future affected nodes noty current ones
        g = deepcopy(self.graph)
        if (len(inf) == 0):
            return 0
        if add_current:
            spread = num_simulations*len(inf)
        else:
            spread = 0

        tasks = [(total_aff, g, inf, spread) for i in range(num_simulations)]
        return parallelize_task(tasks)


class News():  # all news array at t
    # Flags and rel are numpy arrays
    # fake=0 and true=1, eg: self.aff=affected= [[1],[1,2,3],[2,4,6]]
    def __init__(self, tree, affected, rel, labels, flags, beta, gamma, total_aff):
        # in self.flags: flags,fake => 1 & not flags, not fake => 0
        self.tree = tree
        self.aff = affected
        self.labels = labels
        self.rel = rel
        self.flags = flags
        self.beta = beta
        self.gamma = gamma
        self.total_aff = total_aff

    # [[1],[1,2,3],[2,4,6]] => [[3,4],[5,6],[7]], indices of new infected nodes
    def infected(self):
        # flags_allnews: [3,4] flags, [5,6] flags
        new_inf = []
        flags = deepcopy(self.flags)
        k = 0
        for i in self.aff:
            flag_rel = self.rel[:, int(self.labels[k])]
            a = self.tree.affectednodes(i, self.total_aff[k])[0]
            new_inf.append(a)
            random = np.random.random(len(a))
            mult = flag_rel[a] > random
            label = self.labels[k]
            flags[k, a] = np.logical_xor(mult, label)
            k += 1

        return new_inf, flags

    def expected_util(self):  # expected util_t, P_t for all news at t
        P_t = []
        k = 0
        util = np.zeros(self.flags.shape[0])
        for i in self.flags:
            flagged = np.argwhere(i == 1).reshape(-1)
            not_flagged = np.argwhere(i == 0).reshape(-1)
            numerator = np.prod(
                self.rel[flagged, 0])*np.prod(1-self.rel[not_flagged, 0])*(self.beta)
            den = numerator + \
                (1-self.beta)*np.prod(1 -
                                      self.rel[flagged, 1])*np.prod(self.rel[not_flagged, 1])
            prob = numerator/den
            P_t.append(prob)
            k += 1

        for i in range(util.shape[0]):
            util[i] = P_t[i] * \
                self.tree.val_calc(self.aff[i], self.total_aff[i])-self.gamma
        return util, np.array(P_t)  # both numpy arrays


def eff_rel(rel, num):
    if (num % 2 == 0):
        sum = math.comb(num, num//2) * ((1-rel)**(num/2)) * \
            (rel**(num/2)) * 0.5
    else:
        sum = 0

    for i in range(num//2+1, num+1):
        sum += math.comb(num, i) * ((1-rel)**(num-i)) * (rel**(i))
    return sum


def eff_reliability(rel, num):
    if (num % 2 == 0):
        sum = math.comb(num, num//2) * ((1-rel)**(num/2)) * \
            (rel**(num/2)) * 0.5
    else:
        sum = 0

    for i in range(num//2+1, num+1):
        sum += math.comb(num, i) * ((1-rel)**(num-i)) * (rel**(i))

    return sum


class UTIL_min():

    def __init__(self, g, P_t, T, N, total_aff, rel, gamma):
        self.graph = g
        self.gamma = gamma
        self.T = T
        self.N = N
        self.total_aff = total_aff
        self.P_t = P_t
        self.rel = rel
        self.lst = []
        self.prob_lst = []

    # inf=> N= [0,1] => [2,3,4], [0,0,0.5...]
    def val_calc(self, num_simulations=2000, add_current=False):
        # Summed up only future affected nodes noty current ones
        inf = self.N
        total_aff = self.total_aff

        total = deepcopy(total_aff)
        g = deepcopy(self.graph)
        if (len(inf) == 0):
            return 0
        if add_current:
            spread = num_simulations*len(inf)
        else:
            spread = 0
        for _ in range(num_simulations):
            total = deepcopy(total_aff)

            for i in inf:  # this is imp for fut_val_calc, one-step-look ahead
                total[i] = True

            infected = affectednodes(g, inf, total)[0]
            spread += len(infected)
            while (len(infected) > 0):
                for i in infected:
                    total[i] = True
                infected = affectednodes(g, infected, total)[0]
                spread += len(infected)
        return spread / num_simulations

    def calc_min(self, N, total_aff):
        # Need to check current nodes to be added or not
        self.lst = []
        self.recur_min(N, [], 0, len(N), total_aff)
        # [self.lst.insert(pos, elem) for pos, elem in zip(range(1,2*len(self.lst),3), self.lst[0::2])]
        self.fun_min(1, 0, 0)
        return

    def recur_min(self, N, indices, i, maxim, total_aff):
        if (i != maxim):
            ind = deepcopy(indices)
            ind.append(N[i])
            self.recur_min(N, ind, i+1, maxim, total_aff)
            self.recur_min(N, indices, i+1, maxim, total_aff)

        else:
            self.lst.append(self.val_calc())
            return

    def fun_min(self, a, k, i):
        if (k < self.T.shape[0]):
            self.fun_min(a*self.T[k], k+1, i+1)
            self.fun_min(a*(1-self.T[k]), k+1, i)
        else:
            self.prob_lst.append([a, i])
            return

    def fut_calc_min(self):
        sum = 0
        N = self.N
        total_aff = self.total_aff
        P_t = self.P_t
        rel = self.rel
        self.calc_min(N, total_aff)

        for i in range(len(self.lst)):
            prob, numb = self.prob_lst[i]
            eff_rel = eff_reliability(rel, numb)
            p_flagged = 1/(1 + (1/P_t - 1) * (1/eff_rel-1))
            p_unflagged = 1/(1 + (1/P_t - 1) * (eff_rel) / (1-eff_rel))
            term1 = P_t * eff_rel + (1-P_t) * (1-eff_rel)
            term2 = P_t * (1-eff_rel) + (1-P_t) * eff_rel
            sum += prob * ((term1) * max(0, p_flagged * self.lst[i] - self.gamma) + (
                term2) * max(0, p_unflagged * self.lst[i] - self.gamma))
        return sum


class Algorithm_min():
    def __init__(self, news):
        self.news = news  # UTIL(): rel,T,val,gamma as input :rel only for future infected

    def select_news(self):  # Gives news indices to be sent to Oracle
        fut_utils = []
        util_now = self.news.expected_util()
        util_t = np.array(util_now[0])
        P_t = util_now[1]
        util_next = []
        k = 0
        for i in self.news.aff:

            inf = self.news.tree.affectednodes(i, self.news.total_aff[k])[
                1]  # gets prob of infection
            # arguments of future infected nodes with non-zer prob
            inf_arg = np.argwhere((inf > 0) & (
                np.array(self.news.total_aff[k]) == False)).reshape(-1)
            rel = self.news.rel[0][0]
            # P_t, T, N, total_aff, rel, gamma
            U = UTIL_min(self.news.tree.graph, P_t[k], inf[inf_arg],
                         inf_arg, self.news.total_aff[k], rel, self.news.gamma)
            util_next.append(U.fut_calc_min())
            k += 1
        util_next = np.array(util_next)

        return np.argwhere((util_t >= util_next) & (util_t > 0)).reshape(-1)


class UTIL():
    def __init__(self, rel, T, val, gamma):  # Always use this fun --> util_calc
        self.R = rel
        self.T = T
        self.val = val
        self.gamma = gamma
        self.lst = []

    def fun(self, a, b, c, k, n):  # Always use this function 1st
        # kth_user: 0 -> O_f and 1-> O_bar(f) in R[k,0/1]
        if (k == n):
            self.lst.append([a+b, c])
            return
        self.fun(a*self.T[k]*self.R[k, 0], b*self.T[k]*(1-self.R[k, 1]),
                 # flagged by user k+1
                 c*((1-self.R[k, 1])/self.R[k, 0]), k+1, n)
        self.fun(a*self.T[k]*(1-self.R[k, 0]), b*self.T[k] *
                 self.R[k, 1], c*self.R[k, 1]/(1-self.R[k, 0]), k+1, n)
        self.fun(a*(1-self.T[k]), b*(1-self.T[k]), c,
                 k+1, n)  # didn't spread to user k+1

    def util_calc(self):
        sum = 0
        j = 0
        for i in self.lst:
            sum += i[0]*max(0, (1/(1+i[1]))*self.val[j]-self.gamma)
            j += 1
        return sum


class Algorithm():
    def __init__(self, news):
        self.news = news  # UTIL(): rel,T,val,gamma as input :rel only for future infected

    def select_news(self):  # Gives news indices to be sent to Oracle
        fut_utils = []
        util_now = self.news.expected_util()
        util_t = np.array(util_now[0])
        P_t = util_now[1]
        util_next = []
        k = 0
        for i in self.news.aff:
            # G_inf=np.array([self.news.tree.graph.nodes[i]['inf'] for i in self.news.tree.graph.nodes])
            inf = self.news.tree.affectednodes(i, self.news.total_aff[k])[
                1]  # gets prob of infection
            # arguments of future infected nodes with non-zer prob
            inf_arg = np.argwhere((inf > 0) & (
                np.array(self.news.total_aff[k]) == False)).reshape(-1)
            local_rel = self.news.rel[inf_arg]
            fut_val = self.news.tree.fut_val_calc(
                inf_arg, self.news.total_aff[k])  # val for future nodes

            # G_inf=np.array([self.news.tree.graph.nodes[i]['inf'] for i in self.news.tree.graph.nodes])

            U = UTIL(local_rel, inf[inf_arg], fut_val, self.news.gamma)
            U.fun(P_t[k], 1-P_t[k], P_t[k], 0, len(inf_arg))

            util_next.append(U.util_calc())
            k += 1
        util_next = np.array(util_next)

        # if len(np.argwhere((util_next>util_t) & (util_t>0)).reshape(-1)) > 0:
        #     print('Yes!')
        # print(util_next)
        print(f'Algo_Utility Now: {util_t}, Algo_Utility_Look_Ahead: {util_next}')
        return np.argwhere((util_t >= util_next) & (util_t > 0)).reshape(-1)


class Greedy():
    def __init__(self, news):
        self.news = news  # UTIL(): rel,T,val,gamma as input :rel only for future infected

    def select_news(self):  # Gives news indices to be sent to Oracle
        fut_utils = []
        util_now = self.news.expected_util()
        util_t = util_now[0]
        P_t = util_now[1]
        return np.argwhere(util_t > 0).reshape(-1)


class New():
    def __init__(self, news):
        self.news = news  # UTIL(): rel,T,val,gamma as input :rel only for future infected

    def select_news(self):  # Gives news indices to be sent to Oracle
        util_now = self.news.expected_util()
        util_t = util_now[0]
        P_t = util_now[1]
        util_next = []
        k = 0
        for i in self.news.aff:

            inf = self.news.tree.affectednodes(i, self.news.total_aff[k])[
                1]  # gets prob of infection
            # arguments of future infected nodes with non-zer prob
            inf_arg = np.argwhere((inf > 0) & (
                np.array(self.news.total_aff[k]) == False)).reshape(-1)
            local_rel = self.news.rel[inf_arg]
            val_t_1 = self.news.tree.val_next(i, self.news.total_aff[k])
            est_f, est_t = self.news.tree.Estimator(
                P_t[k], inf[inf_arg], local_rel)
            utility = P_t[k]*(np.heaviside(est_f*val_t_1 - self.news.gamma, 0)*(val_t_1 - self.news.gamma)) + (
                1-P_t[k])*(np.heaviside(est_t*val_t_1 - self.news.gamma, 0)*(-self.news.gamma))
            util_next.append(utility)
            k += 1
        print(f'Approx_Utility Now: {util_t}, Approx_Utility_Look_Ahead: {util_next}')
        return np.argwhere((util_t >= util_next) & (util_t > 0)).reshape(-1)


def generate_cayley_graph(group, generators, order, w):
    G = nx.Graph()
    elements = list(range(order))

    # Add nodes
    G.add_nodes_from(elements)

    # Add edges based on group generators
    for element in elements:
        for generator in generators:
            neighbor = (element * generator) % order
            G.add_edge(element, neighbor, wt=w)

    return G


def generate_constant_weighted_karate_club_graph(weight):
    """
    Generates a Karate Club graph with constant edge weights.

    Parameters:
        - weight (int): The constant weight for all edges.

    Returns:
        - G (NetworkX graph): The generated Karate Club graph.
    """
    G = nx.karate_club_graph()

    # Add constant weight to all edges
    for u, v in G.edges():
        G[u][v]['wt'] = weight

    return G
