from Classes import *
# import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_constant_weighted_watts_strogatz_graph(num_nodes, nearest_neighbors, rewiring_probability, weight):
    """
    Generates a Watts-Strogatz small-world graph with constant edge weights.

    Parameters:
        - num_nodes (int): The number of nodes in the graph.
        - nearest_neighbors (int): The number of nearest neighbors to connect to each node initially.
        - rewiring_probability (float): The probability of rewiring each edge.
        - weight (int): The constant weight for all edges.

    Returns:
        - G (NetworkX graph): The generated Watts-Strogatz graph.
    """
    G = nx.Graph()


    # Add nodes
    G.add_nodes_from(range(num_nodes))

    # Connect each node to its nearest neighbors with constant weight
    for node in range(num_nodes):
        for i in range(1, nearest_neighbors + 1):
            neighbor = (node + i) % num_nodes  # Right neighbor
            G.add_edge(node, neighbor, wt=weight)

            neighbor = (node - i) % num_nodes  # Left neighbor
            G.add_edge(node, neighbor, wt=weight)

    # Rewire edges with probability p
    for u, v in list(G.edges()):
        if random.random() < rewiring_probability:
            new_v = random.choice(range(num_nodes))
            while new_v == u or G.has_edge(u, new_v):
                new_v = random.choice(range(num_nodes))
            G.remove_edge(u, v)
            G.add_edge(u, new_v, wt=weight)

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

if __name__=='__main__':
    # Parameters
    num_nodes = 20
    nearest_neighbors = 2
    rewiring_probability = 0.2
    weight = 0.7  # Constant weight for all edges

    # Generate the constant weighted Watts-Strogatz graph
    ws_graph = generate_constant_weighted_watts_strogatz_graph(num_nodes, nearest_neighbors, rewiring_probability, weight)

    # Draw the graph
    pos = nx.spring_layout(ws_graph)  # You can use other layouts as well
    nx.draw(ws_graph, pos, with_labels=True, node_color='lightblue', node_size=800)

    # Extract edge weights
    edge_labels = {(u, v): ws_graph[u][v]['wt'] for u, v in ws_graph.edges()}

    # Draw edge labels
    nx.draw_networkx_edge_labels(ws_graph, pos, edge_labels=edge_labels)

    plt.title("Constant Weighted Watts-Strogatz Small-World Graph")
    plt.show()