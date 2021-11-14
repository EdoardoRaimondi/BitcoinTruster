# File cointaining usefull functions:
#    drawgraph() -> used to draw a graph with some characteristics
#    min_max()   -> used to have min and max
import matplotlib.pyplot as plt
import networkx as nx
import sys
import math
from networkx.classes.graph import Graph

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

def drawsubgraph(graph, node, type):
    # param    (networkx graph) : directed graph
    # param       (integer)     : value of a node
    # param       (integer)     : type of graph we want, for now possible values are:
    #                              -- 1 : draw a graph with entering neighbors of the given node
    #                              -- 2 : draw a graph with outgoing neighbors from the given node
    #                              -- 3 : draw a graph with all the neighbors of the given node
    # raises networkxerror if node not in the graph
    # draw a wanted subgraph dipends on the type subgraph

    # preliminary check
    if 0 < type and type < 4:
        # initialize the subgraph
        subgraph = nx.DiGraph()
        subgraph.add_node(node)

        if type == 1:
            subnodes = graph.predecessors(node)
            for previous_node in subnodes:
                subgraph.add_edge(previous_node, node)
            figure1 = plt.subplot(121)
            nx.draw(subgraph, with_labels=True)
            
        if type == 2:
            subnodes = graph.successors(node)
            for successor_node in subnodes:
                subgraph.add_edge(node, successor_node)
            figure1 = plt.subplot(121)
            nx.draw(subgraph, with_labels=True)
            
        if type == 3:
            subnodes_predecessor = graph.predecessors(node)
            for previous_node in subnodes_predecessor:
                subgraph.add_edge(previous_node, node)

            subnodes_successor = graph.successors(node)
            for successor_node in subnodes_successor:
                subgraph.add_edge(node, successor_node)
            figure1 = plt.subplot(121)
            nx.draw(subgraph, with_labels=True)

#def drawGraph_Centrality():


def drawGraphGoodFair(nodes_goodness, nodes_fairness, number):
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


