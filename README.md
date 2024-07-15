# Rumor Detection
The goal of this project is to reduce the spread of misinformation in social networks by using crowd signals. We independently _learn_ the reliability of users to make effective use of flags, while also using the social network graph to identify and eliminate rumors from the vast, dynamically generated news within the network.
## Problem Setting
We employ an Independent Cascade Model (ICM) to simulate the spread of information in the network with known parameters. Each user in the network, once exposed to a piece of news, has the option to flag it. 
There is also a cost associated with each user being exposed to fake news. We have an Oracle that can provide the true label of any news item, with no limit on the number of news items submitted for verification at any given time. 
However, each verification incurs a cost.


An algorithm must select which news items to query the Oracle at each time step from a vast pool of news, aiming to minimize the total cost incurred during querying while potentially allowing some fake news to spread to users. 
An efficient algorithm must utilize both the crowd signals and the network structure to decide which set of news items to query at each time step.
