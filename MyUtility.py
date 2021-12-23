# File cointaining usefull functions:
#    drawgraph() -> used to draw a graph with some characteristics
#    min_max()   -> used to have min and max
import matplotlib.pyplot as plt
import networkx as nx
import sys
import math
import numpy as np
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
            labels_dict[node] = "Node id: {}, \nFairness: {}".format(node, float16(values_nodes[node]))

    # create the subgraph
    for node1 in list(nodes):
        for node2 in list(nodes)[1:]:
            if graph.has_edge(node1, node2):
                subgraph.add_edge(node1, node2)
    
    # print out the graph created
    plt.title("Subgraph")
    nx.draw(subgraph, with_labels=True, labels=labels_dict)
    plt.show()

def draw_histogram(graph, type):
    # param   (networkx graph) : directed graph
    # param      (integer)     : type of values that we are looking for, possible values:
    #                            -- 0 we want a histogram of edges
    #                            -- 1 we want a histogram with global degree
    #                            -- 2 we want a histogram with in degree and out degree
    # draw a wanted subgraph dipends on the type subgraph

    if type == 0:
        # we want the edges histogram
        weights = np.array(list(nx.get_edge_attributes(graph, 'weight').values())) # contains all weights
        unique, counts = np.unique(weights, return_counts=True)
        count = dict(zip(unique, counts))
        x_label = "edge weights"
        title = "Edge weights histogram"
        width = 0.3
    elif type == 1:
        # we want the global degree histogram
        x_in = list(dict(graph.in_degree(graph.nodes())).values())
        x_out = list(dict(graph.out_degree(graph.nodes())).values())
        x = np.array(x_in + x_out)
        unique, counts = np.unique(x, return_counts=True)
        count = dict(zip(unique, counts))
        x_label = "degree"
        title = "Generale degree (in degree + out degree) histogram"
        width = 1.5
    elif type == 2:
        # we want the histogram of in degree and out degree
        x_in = np.array(list(dict(graph.in_degree(graph.nodes())).values()))
        x_out = np.array(list(dict(graph.out_degree(graph.nodes())).values()))
        unique, counts = np.unique(x_in, return_counts=True)
        count_in = dict(zip(unique, counts))
        unique, counts = np.unique(x_out, return_counts=True)
        count_out = dict(zip(unique, counts))
        in_title = "In degree histogram"
        out_title = "Out degree histogram"
        width = 1.5
        
        # plots
        fig, axs = plt.subplots(2)
        axs[0].bar(list(count_in.keys()), list(count_in.values()), width=width, align='center', color='b')
        axs[0].set_title(in_title)
        axs[0].set(ylabel="count", xlabel="degree")
        axs[1].bar(list(count_out.keys()), list(count_out.values()), width=width, align='center', color='r')
        axs[1].set_title(out_title)
        axs[1].set(xlabel="degree", ylabel="count")
        fig.tight_layout()
        plt.show()
        return
    
    # prepare plot for single histogram
    if type == 0 or type == 1:

        # plot the results  
        plt.bar(list(count.keys()), list(count.values()), width=width, align='center', color='blue')
        plt.xlabel(x_label)
        plt.ylabel("count")
        plt.title(title)

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

def draw_graph_scatter(list_all_nodes, dict_first, dict_second, name_first, name_second, title):
    # param    (list)  : list of all nodes
    # param    (dict)  : dict key-value as node-goodness
    # param    (dict)  : dict key-value as node-fairness
    # param  (string)  : name of the first dict passed 
    # param  (string)  : name of the second dict passed 
    # param  (string)  : title of what printed out
    # plot and show the scatter graph with also the outlinears
    nodes_plot = {}
    for node in list_all_nodes:
        if node in dict_first.keys() and node in dict_second.keys():
            nodes_plot[node] = [dict_first[node], dict_second[node]]
    
    # plot the nodes
    for node in nodes_plot.keys():
        plt.scatter(nodes_plot[node][0], nodes_plot[node][1], color='k')
    
    plt.title(title)
    plt.xlabel(name_first)
    plt.ylabel(name_second)
    plt.show()

def get_node_features(list_all_nodes, first_dict, second_dict):
    # param   (list)   : list of all keys that we are looking for
    # param   (dict)   : first dict that has node-goodness as key-value
    # param   (dict)   : second dict that has node-fairness as key-value
    # function that given 2 dicts and a list of interested keys, return a dict with keys that stay in bot dict passed and
    # values the 2 values of the dict passed. 
    nodes_features = {}
    for node in list_all_nodes:
        if node in first_dict.keys() and node in second_dict.keys():
            if first_dict[node] != 0: # remove outlinears
                nodes_features[node] = [first_dict[node], second_dict[node]]
    return nodes_features

def ranking(list_all_nodes, first_dict, second_dict, damping_factor):
    # param  (list)  : list of all key
    # param  (dict)  : first dict where we found the first value
    # param  (dict)  : second dict where we found the second value
    # param  (int)   : value of a dampling factor to recalibrate the scores
    # return (dict)  : returna a dict key-value as node-score
    node_value = {}
    for node in list_all_nodes:
        if node in first_dict.keys() and node in second_dict.keys():
            node_value[node] = damping_factor*(first_dict[node] - second_dict[node])
    
    sort_dict = dict(sorted(node_value.items(), key=lambda x: x[1], reverse=True))
    return sort_dict
