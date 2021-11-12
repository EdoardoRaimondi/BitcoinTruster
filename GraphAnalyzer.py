import time
import sys
import networkx as nx
import pandas as pd
import numpy as np
import MyUtility
from networkx.classes import graph
from networkx.algorithms.centrality.closeness import closeness_centrality
from networkx.algorithms.centrality.betweenness import betweenness_centrality
from networkx.classes.function import is_directed, number_of_nodes, subgraph, degree

class GraphAnalyzer:
    # Wrapper class for the analyzing methods
    
    def __init__(self, graph):
        # param : graph to analyze
        self.graph = graph

    def graph_in_degree(self):
        # Calculate degree of all node in the graph
        # param  (networkx graph) : directed graph
        # returns  (dict), (int)  : dict with key as node ad degree as value
        # Print out also the exectution time and the node with max degree, node with max degree

        start_time = time.monotonic()
        degree_nodes = dict(self.graph.in_degree(self.graph.nodes)) # need a conversion tu use always the same lambda function
        end_time = time.monotonic()
        max_degree_node = max(degree_nodes, key = lambda x: degree_nodes[x])
        print("----DEGREE ANALYSIS----")
        print("max degree node: {}, degree: {}".format(max_degree_node, degree_nodes[max_degree_node]))
        print("execution time: {}".format(end_time-start_time))

        return degree_nodes, max_degree_node

    def graph_centrality(self):
        # Calculate degree of all node in the graph
        # param  (networkx graph) : directed graph
        # returns  (dict), (int)  : dict with key as node ad closeness centrality as value, node with max closeness centrality
        # Print out also the exectution time and the node with max closeness centrality

        start_time = time.monotonic()
        centrality_nodes = closeness_centrality(self.graph)
        end_time = time.monotonic()
        max_centrality_node = max(centrality_nodes, key = lambda x: centrality_nodes[x])
        print("----CLOSENESS CENTRALITY ANALYSIS----")
        print("max closeness node: {}, closeness: {}".format(max_centrality_node, centrality_nodes[max_centrality_node])) 
        print("execution time: {}".format(end_time-start_time)) #my pc 90sec

        return centrality_nodes, max_centrality_node

    def graph_betweenness(self):
        # Calculate degree of all node in the graph
        # param  (networkx graph) : directed graph
        # returns  (dict), (int)  : dict with key as node ad between centrality as value, node with max betweenness centrality
        # Print out also the exectution time and the node with max betweenness centrality
        
        start_time = time.monotonic()
        betweenness_nodes = betweenness_centrality(self.graph)
        end_time = time.monotonic()
        max_betweennes_node = max(betweenness_nodes, key = lambda x: betweenness_nodes[x])
        print("----BETWEENNESS CENTRALITY ANALYSIS----")
        print("max betweenness node: {}, betweenness: {}".format(max_betweennes_node, betweenness_nodes[max_betweennes_node]))  
        print("execution time: {}".format(end_time-start_time)) #my pc 196sec

        return betweenness_nodes, max_betweennes_node

    def node_goodness(self, node):
        # Calculates goodnees of a node
        # param   (netowrkx graph) : directed graph
        # param        (int)       : node_id
        # returns     (double)     : node goodnees score. None if there are no weights
        # raises  NetowrkXError if the node is not in the graph

        if self.graph.in_degree(node) == 0:
            return # no grade available for this node
        weights = []
        for n in self.graph.predecessors(node): # I want the entering nodes 
            weights.append(self.graph[n][node]["weight"])

        recalibration_factor = np.log(len(weights)) # for score recalibration purposes ( see "paper" for more details)
        # NB : node with only 1 grade will not be considered relevant enough, then their goodness score is 0

        return (sum(weights)/self.graph.in_degree(node))*recalibration_factor

    def graph_goodness(self, nodes_number=None):
        # Calculate goodnees of all the nodes of the graph
        # param graph (directed networkx graph) 
        # param nodes_number  (int)    : number of nodes to consider.
        #                                If no values is provided, the entire graph will be considered.
        # return (dict), (int), (int)  : dict key - value as node ID - goodness, node with max goodness, 
        #                                node with min goodness

        start_time = time.monotonic()
        if nodes_number is None:
            nodes_number = self.graph.number_of_nodes()

        remaining_nodes = nodes_number

        node_in_degree_dict = {}
        for node, degree in self.graph.in_degree(self.graph.nodes):
            # create a proper dictionary instead of a "DegreeView" as provided by in_degree networkx method
            node_in_degree_dict[node] = degree

        # finally return the goodnees of our nodes
        nodes_goodnesses = {}
        for node in node_in_degree_dict.keys():
            remaining_nodes -= 1
            if self.node_goodness(node) != None :
               nodes_goodnesses[node] = self.node_goodness(node)
            if remaining_nodes == 0 : # I'm done
                end_time = time.monotonic()
                print("----GOODNESS ANALYSIS----")
                print("exectution time for {} nodes : {}".format(nodes_number, end_time-start_time))
                # look to the node with greater and lower goodness
                node_min_goodness, node_max_goodness = MyUtility.min_max(nodes_goodnesses)
                print("best node: {} has goodness: {}".format(node_max_goodness, nodes_goodnesses[node_max_goodness]))
                print("worst node: {} has goodness: {}".format(node_min_goodness, nodes_goodnesses[node_min_goodness]))

                return nodes_goodnesses, node_max_goodness, node_min_goodness

    # postit comment: nodes_goodness are with the ln(in_degree(node)) -> I diveded by it (be carefull on this)
    def node_fairness(self, node, nodes_goodness):
        # calculate the fariness of a node
        # param graph (directed networkx graph) 
        # param node          (int)        : number of nodes to consider.
        # parma nodes_goodness    (dict)   : dict key-value as node-goodness_value
        # return         (double)          : return the value of the fairness
        # raises presonalized errors

        # check if the node has successors, if not return 0
        if self.graph.out_degree(node) == 0:
            return 0

        # create a dict with all successor of such node and their goodness score
        nodes_successors_and_goodness = {}
        for successor in self.graph.successors(node):
            if self.graph.in_degree(successor) == 1: # if the successor has only this node it is useless
                nodes_successors_and_goodness[successor] = None
            else:
                nodes_successors_and_goodness[successor] = (nodes_goodness[successor]) / np.log(self.graph.in_degree(successor))

        # create a dict with all successor of such node and the evaluation give to them by the node
        nodes_successors_and_evaluation = {}
        for successor in self.graph.successors(node):
            nodes_successors_and_evaluation[successor] = self.graph[node][successor]["weight"]

        # remove useless successor such they that are only the one with None as value 
        for n in list(nodes_successors_and_goodness.keys()):
            if nodes_successors_and_goodness[n] is None:
                del(nodes_successors_and_goodness[n])
                del(nodes_successors_and_evaluation[n])

        # check if the len of the 2 dict are the same
        if len(nodes_successors_and_goodness) != len(nodes_successors_and_evaluation):
            sys.exit("[ERROR] Different value on dictionary used to calcluate fairness")

        # calculate the average variance on the evaluation
        # notice the abs() l1 norm if we want we can pass to l2 metrics
        variance = 0
        for n in nodes_successors_and_goodness.keys(): 
            variance = variance + abs((abs(nodes_successors_and_evaluation[n]) - abs(nodes_successors_and_goodness[n])))
        
        return variance / len(list(self.graph.successors(node)))

    def graph_fariness(self, node_goodness):
        # calculate the fariness of all nodes
        # param graph (directed networkx graph) 
        # parma nodes_goodness    (dict)   : dict key-value as node-goodness_value
        # return   (dict), (int), (int)    : return a dict key-value as node-fairness, node with min and max fairness
        # raises presonalized errors

        print("----FAIRNESS ANALYSIS----")
        print("The scale is between 0 and 10, if 0 good if 10 bad")
        nodes_fairness = {}
        start_time = time.monotonic()

        for node in self.graph.nodes():
            fairness = self.node_fairness(node, node_goodness)
            nodes_fairness[node] = fairness
        
        end_time = time.monotonic()

        # remove all nodes that has 0 out edges
        for node in self.graph.nodes():
            if self.graph.out_degree(node) == 0:
                del(nodes_fairness[node])

        min_node_fariness, max_node_fairness = MyUtility.min_max(nodes_fairness)

        print("exectution time : {}".format(end_time-start_time))
        print("worst node: {} has fairness: {}".format(max_node_fairness, nodes_fairness[max_node_fairness]))
        print("best node: {} has fairness: {}".format(min_node_fariness, nodes_fairness[min_node_fariness]))

        
        return nodes_fairness, min_node_fariness, max_node_fairness
        