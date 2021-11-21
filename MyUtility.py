# File cointaining usefull functions:
#    drawgraph() -> used to draw a graph with some characteristics
#    min_max()   -> used to have min and max
import matplotlib.pyplot as plt
import networkx as nx
import sys
import math
from networkx.classes.graph import Graph
from numpy import float16

def weighted_incoming_mean(graph, node):
    # calculate the weighted mean of incoming edges of input node on the input graph
    if not graph.has_node(node):
        return
    weights = []
    for n in graph.predecessors(node): # I want the entering nodes 
        weights.append(graph[n][node]["weight"])
    return sum(weights)/graph.in_degree(node)

def min_max(dict):
    # param     (dict)     : dictionary key-value
    # return (int), (int)  : min and max value in the key

    min = math.inf
    max = -math.inf
    node_max, node_min = None, None
    for key in dict.keys():
        if dict[key] > max:
            node_max = key
            max = dict[key]
        if dict[key] < min:
            node_min = key
            min = dict[key]
    return node_min, node_max

def draw_subgraph(graph, nodes, values_nodes, type):
    # param   (networkx graph) : directed graph
    # param       (list)      : list with the id of the nodes that we want to print out
    # param       (dict)      : dict key-values as node id-values
    # param     (integer)     : type of graph we want, for now possible values are:
    #                              -- 1 : draw a graph with goodness as values passed
    #                              -- 2 : draw a graph with fairness as values passed
    # draw a wanted subgraph dipends on the type subgraph

    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(nodes)

    # calculate the labels to print
    labels_dict = {}
    for node in nodes:
        if type == 1:
            labels_dict[node] = "Node id: {}, \nGoodness: {}".format(node, float16(values_nodes[node]))
        else:
            labels_dict[node] = "Node id: {}, \Fairness: {}".format(node, float16(values_nodes[node]))

    # create the subgraph
    for node1 in list(nodes):
        for node2 in list(nodes)[1:]:
            if graph.has_edge(node1, node2):
                subgraph.add_edge(node1, node2)
    
    # print out the graph created
    plt.title("Subgraph")
    nx.draw(subgraph, with_labels=True, labels=labels_dict)
    plt.show()

def draw_graph_centrality(degree_nodes, centrality_nodes, betweenness_nodes, number):
    # param    (dict)  : dict key-value as node-degree
    # param    (dict)  : dict key-value as node-closeness centrality
    # param    (dict)  : dict key-value as node-betweenness centrality
    # param    (int)   : number of nodes that we want to see in the graph
    # draw a graph with some nodes and their the values of degree, closeness centrality and betweenness centrality

    # sort the dicts
    sorted_nodes_degree = dict(sorted(degree_nodes.items()))
    sorted_nodes_centrality = dict(sorted(centrality_nodes.items()))
    sorted_nodes_betweenness = dict(sorted(betweenness_nodes.items()))

    # plot the values
    plt.plot(list(sorted_nodes_degree.keys())[:number], list(sorted_nodes_degree.values())[:number], label='Degree', marker="s")
    plt.plot(list(sorted_nodes_centrality.keys())[:number], list(sorted_nodes_centrality.values())[:number], label='Closeness Centrality', marker="s")
    plt.plot(list(sorted_nodes_betweenness.keys())[:number], list(sorted_nodes_betweenness.values())[:number], label='Betweenness Centrality', marker="s")
    plt.xlabel('Node')
    plt.ylabel('value')
    plt.title('Degree-Closeness Centrality-Between Centrality for each node')
    plt.legend()

    # show the graph
    plt.show()


def draw_graph_good_fair(nodes_goodness, nodes_fairness, number):
    # param    (dict)  : dict key-value as node-goodness
    # param    (dict)  : dict key-value as node-fairness
    # param    (int)   : number of nodes that we want to see in the graph
    # raises personalized error if number is too large
    # draw a graph with some nodes and their the values of goodness and fairness

    # sort the dicts
    sorted_nodes_goodness = dict(sorted(nodes_goodness.items()))
    sorted_nodes_fairness = dict(sorted(nodes_fairness.items()))

    # take values that stay in both dicts
    # by definition always fairness dict is smaller than goodness dict 
    keys_used = []
    for key in sorted_nodes_fairness.keys():
        if key in sorted_nodes_goodness.keys():
            keys_used.append(key)    

    # check if there are at least the nodes required to print
    if number > len(keys_used):
        sys.exit("[ERROR] number is too large")

    # create the values for goodness and fairness for the nodes required
    y_goodness, y_fairness = [], []
    for key in keys_used[:number]:
        y_goodness.append(sorted_nodes_goodness[key])
        y_fairness.append(sorted_nodes_fairness[key])


    # plot the values
    plt.plot(keys_used[:number], y_goodness, label='Goodness', marker="s")
    plt.plot(keys_used[:number], y_fairness, label='Fairness', marker="s")
    plt.xlabel('Node')
    plt.ylabel('value')
    plt.title('Goodness-fairness for each node')
    plt.legend()

    # show the graph
    plt.show()
