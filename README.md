# Rumor Detection
The goal of this project is to reduce the spread of misinformation in social networks by using crowd signals. We independently _learn_ the reliability of users to make effective use of flags, while also using the social network graph to identify and eliminate rumors from the vast, dynamically generated news within the network.
## Problem Setting
We employ an Independent Cascade Model (ICM) to simulate the spread of information in the network with known parameters. Each user in the network, once exposed to a piece of news, has the option to flag it. 
There is also a cost associated with each user being exposed to fake news. We have an Oracle that can provide the true label of any news item, with no limit on the number of news items submitted for verification at any given time. 
However, each verification incurs a cost.


An algorithm must select which news items to query the Oracle at each time step from a vast pool of news, aiming to minimize the total cost incurred during querying while potentially allowing some fake news to spread to users. 
An efficient algorithm must utilize both the crowd signals and the network structure to decide which set of news items to query at each time step.
# Code Usage Guide

We are simulating an environment where we are comparing the performance of Greedy Algo with our _One-Step Look Ahead Algo_ where both the algorithms are aware of user reliabilities and the social network structure.

## Classes.py

Contains all the necessary functions and Classes used to run `execute.py`.

## graph.py

`graph.py` contains the functions used to simulate graphs such as __Watts-Strogatz Graph__ and __Zachary’s Karate Club Graph__.

## execute.py

This is the file that simulates the task of performance evaluation for both the algos evaluated on the same sample path. Here sample path refers to the users affected and their respective flags for each news. `execute.py` at every time step generates news items from 2 randomly selected nodes in the graph and the labels for these news are given by the source reliabilities of the nodes, known to the network. At every time step each algo selects the news items to be queryied and we run evaluate the algorithm for a default value of 10 time-steps running over a 100 itterations (`iter`=100) for comparing the performance. `execute.py` operates on a constant weighted Watts-Strogatz Graph `ws_graph`. The parameters include the `num_nodes`: number of nodes, `nearest_neighbors`: number of nearest neighbors, `rewiring_probability`: rewiring probability, and `weight`:weight of each link.

## Parameters

- `beta`: Represents the source unreliability.
- `gamma`: Cost of querying relative to the cost of a user being affected by fake news.
- `rel`: List of flagging accuracies of users in the network where each element is a list of $\theta_{u, f}$ and $\theta_{u, \overline{f}}$.
- `u1`: Utility of the Opt Algorithm (algorithm that knows the true label of news and decides whether to query as soon as news generated) as a function of the number of steps, averaged over all iterations.
- `u_2`, `u_4`: Utilities of the One-step Look Ahead Algorithm and the Greedy Algorithm, respectively, as functions of the number of steps, averaged over all iterations.


## Binary Tree.ipynb

`Binary Tree.ipynb` is a standalone Jupyter Notebook file dedicated to evaluating the performance of the algorithms on binary trees.

