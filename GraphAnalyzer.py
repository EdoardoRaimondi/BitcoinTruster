import itertools
from operator import truediv
import time
import sys
import math
import threading
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import levene
import scipy
import MyUtility
import ThreadingGraphAnalyzer
import matplotlib.pyplot as plt
from itertools import combinations
from networkx.classes import graph
from networkx.algorithms.centrality.closeness import closeness_centrality
from networkx.algorithms.centrality.betweenness import betweenness_centrality
import multiprocessing
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
        # Calculate centrality of all node in the graph
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
        # Calculate betweennes of all node in the graph
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

        in_degree = self.graph.in_degree(node)
        if in_degree == 0:
            return # no grade available for this node

        recalibration_factor = np.log(in_degree) # for score recalibration purposes ( see "paper" for more details)
        # NB : node with only 1 grade will not be considered relevant enough, then their goodness score is 0

        return (MyUtility.weighted_incoming_mean(self.graph, node))*recalibration_factor

    def graph_goodness(self, nodes_number=None):
        # Calculate goodnees of a certain number of nodes in the graph
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
    def node_fairness(self, node):
        # calculate the fairness of a node
        # param graph (directed networkx graph) 
        # param node          (int)        : number of nodes to consider
        # return         (double)          : return the value of the fairness
        # raises presonalized errors

        # check if the node has successors, if not return 0
        if self.graph.out_degree(node) == 0:
            return 0

        # create a dict with all successor of such node and their feedback scores (weighted mean of incoming edges)
        nodes_successors_and_feedback_score = {}
        for successor in self.graph.successors(node):
            if self.graph.in_degree(successor) == 1: # if the successor has only this node it is outliner
                nodes_successors_and_feedback_score[successor] = None
            else:
                nodes_successors_and_feedback_score[successor] = MyUtility.weighted_incoming_mean(self.graph, successor) 
    
        # create a dict with all successors of such node and the evaluation give to them by the node
        nodes_successors_and_evaluation = {}
        for successor in self.graph.successors(node):
            nodes_successors_and_evaluation[successor] = self.graph[node][successor]["weight"]

        # remove useless successor such they that are only the one with None as value 
        for n in list(nodes_successors_and_feedback_score.keys()):
            if nodes_successors_and_feedback_score[n] is None:
                del(nodes_successors_and_feedback_score[n])
                del(nodes_successors_and_evaluation[n])

        # check if the len of the 2 dict are the same
        if len(nodes_successors_and_feedback_score) != len(nodes_successors_and_evaluation):
            sys.exit("[ERROR] Different value on dictionary used to calculate fairness")

        # calculate the average varsiance on the evaluation
        # notice the abs() l1 norm if we want we can pass to l2 metrics
        variance = 0
        for n in nodes_successors_and_feedback_score.keys(): 
            variance = variance + abs((abs(nodes_successors_and_evaluation[n]) - abs(nodes_successors_and_feedback_score[n])))
        
        return variance / len(list(self.graph.successors(node)))

    def graph_fairness(self):
        # calculate the fairness of all nodes
        # param graph (directed networkx graph) 
        # parma nodes_goodness    (dict)   : dict key-value as node-goodness_value
        # return   (dict), (int), (int)    : return a dict key-value as node-fairness, node with min and max fairness
        # raises presonalized errors

        print("----FAIRNESS ANALYSIS----")
        print("The scale is between 0 and 10, if 0 good if 10 bad")
        nodes_fairness = {}
        start_time = time.monotonic()

        for node in self.graph.nodes():
            fairness = self.node_fairness(node)
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

        # we return less node bc we removed node that has only 1 successor or 0 successor
        return nodes_fairness, min_node_fariness, max_node_fairness

    def are_transations_casual(self, alpha):
        # Hypotesis testing
        # HO : there is indipendence between nodes connections ( transanctions are casual ) 
        # H1 : there is correlation between nodes connetions   ( transaction are not casual, i.e. better users tend to receive more transactions )
        # alpha (float) : confidence level
        # return (boolean) : true if the transactions are casual, false if there are some correlations among them

        rand_graph_number = 10
        degree_sequence = sorted([d for n, d in self.graph.degree()], reverse = True)
        original_degree_distribution = np.unique(degree_sequence, return_counts = True)[1]# here I have (sorted degree values array)(number of nodes with the corresponding degree value)
        #let's consider the second array variance and compare same result on some random graph with the same characteristics of the original one

        p_values = []
        for i in range(rand_graph_number):
          #random graph generation
          r_g = nx.erdos_renyi_graph(n=self.graph.number_of_nodes(), p=((self.graph.number_of_edges()/scipy.special.binom(self.graph.number_of_nodes(),2))), seed=None, directed=True) # p is set such that I have the same number of edges in expectation
          degree_sequence = sorted([d for n, d in r_g.degree()], reverse = True)
          r_g_degree_distribution = np.unique(degree_sequence, return_counts = True)[1]# here I have (sorted degree values array)(number of nodes with the corresponding degree value)
          #let's consider the second array variance and compare same result on some random graph with the same characteristics of the original one
          p_values.append(levene(original_degree_distribution, r_g_degree_distribution))

        for stat in p_values:
            if stat[1] > alpha:
                return True # if p value is more than alpha, the null hp is likely to happen
        return False # all p values are less than alpha, null hp is unlikely to happen
        
    def subgraph_goodness(self, goodness_nodes, number):
        # calclulate the subgraph with grater goodness
        # param graph (directed networkx graph) 
        # parm     goodness_nodes   (dict)   : dict key-value as node-goodness_value
        # parm        number      (int)      : number of nodes that the subgraph must has
        # return           (tuple)           : return the tuple with the greater goodness 

        print("Search the subgraph with {} nodes that it has the higher goodness".format(number))

        # parameters
        max_goodness = -math.inf
        final_nodes_id = []   
        start_time = time.monotonic()

        # see all possibile combinations
        for nodes in combinations(self.graph.nodes, number):
            subgraph = self.graph.subgraph(nodes)

            # check if it is connected
            if nx.is_weakly_connected(subgraph):
                subgraph_goodness = 0
                for node in nodes:

                    # check if it has a goodness value, remember some value hasn't it
                    if node in goodness_nodes.keys():
                        subgraph_goodness = subgraph_goodness + goodness_nodes[node]

                # check if we need to update the values
                if subgraph_goodness > max_goodness:
                    max_goodness = subgraph_goodness
                    final_nodes_id = nodes

        end_time = time.monotonic()
        print("execution time: {}".format(end_time-start_time))

        return final_nodes_id

    def subgraph_fairness(self, fairness_nodes, number):
        # calclulate the subgraph with grater goodness
        # param graph (directed networkx graph) 
        # parm     goodness_nodes   (dict)   : dict key-value as node-goodness_value
        # parm        number      (int)      : number of nodes that the subgraph must has
        # return           (tuple)           : return the tuple with the greater goodness 

        print("Search the subgraph with {} nodes that it has the lower fairness".format(number))

        # parameters
        min_fairness = math.inf
        final_nodes_id = []   
        start_time = time.monotonic()

        # see all possibile combinations
        for nodes in combinations(self.graph.nodes, number):
            subgraph = self.graph.subgraph(nodes)

            # check if it is connected
            if nx.is_weakly_connected(subgraph):
                subgraph_fairness = 0
                for node in nodes:

                    # check if it has a goodness value, remember some value hasn't it
                    if node in fairness_nodes.keys():
                        subgraph_fairness = subgraph_fairness + fairness_nodes[node]

                # check if we need to update the values
                if subgraph_fairness < min_fairness:
                    min_fairness = subgraph_fairness
                    final_nodes_id = nodes

        end_time = time.monotonic()
        print("execution time: {}".format(end_time-start_time))

        return final_nodes_id


#Multiprocessing it's managed by the operation system
    def paralelSubgraph(self, fairness_nodes,num_processor,number):
        if(num_processor == 0):#We cannot have the number of processor equalt to zero! Or we will use the other mathod!
            pass
        combination_nodes = list()
        num_nodes = len(list(combination_nodes))
        print("--------")
        print(num_nodes)
        print("--------")
        for elem in combinations(self.graph.nodes, number):
            combination_nodes.append(elem)
        num_nodes = (len(combination_nodes))
        print("Len of slice graph!")
        print(num_nodes)
        slice_number = round((num_nodes/num_processor)+1)
        if(slice_number > number):
            pass
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        start_time = time.monotonic()
        for i in range(num_processor):
            slice_graph = list(itertools.islice(combination_nodes,(i)*slice_number,(i+1)*slice_number, 1))
            print("Slice:")
            print(slice_graph[2])
            p = multiprocessing.Process(target=ThreadingGraphAnalyzer.worker, args=(i, return_dict, self.graph, fairness_nodes, slice_graph))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()#We need to wait that all the process are finished :)
        min = math.inf
        end_time = time.monotonic()
        print("-----Time------")
        print(end_time-start_time)
