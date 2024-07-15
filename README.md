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

This is the file that simulates the task of performance evaluation for both the algos evaluated on the same sample path. Here sample path refers to the users affected and their respective flags for each news. `execute.py` at every time step generates news items from 2 randomly selected nodes in the graph and the labels for these news are given by the source reliabilities of the nodes, known to the network. At every time step each algo selects the news items to be queryied and we run evaluate the algorithm for a default value of 10 time-steps running over a 100 itterations for comparing the performance.

