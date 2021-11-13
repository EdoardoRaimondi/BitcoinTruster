# File cointaining usefull functions:
#    drawgraph() -> used to draw a graph with some characteristics
#    min_max()   -> used to have min and max
import matplotlib.pyplot as plt
import networkx as nx
import math
from networkx.classes.graph import Graph


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