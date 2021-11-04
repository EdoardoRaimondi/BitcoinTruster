import time
from networkx.classes import graph 
import numpy as np

class GraphAnalyzer:
    # Wrapper class for the analyzing methods
    
    def __init__(self, graph):
        # param : graph to analyze
        self.graph = graph

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
        # return dictionary key - value as node ID - goodness

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
                print("TIME FOR {} NODES GOODNESS SCORE CALCULATION : {}".format(nodes_number, end_time-start_time))
                return nodes_goodnesses
