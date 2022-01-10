import itertools
import math
import time
from itertools import combinations
from threading import Thread
import networkx as nx

#This class it is the non vanilla Thread of python , because if we will use the vanilla class we cannot return nothing when the thread is end!
#So for this scope, i have extended the vanilla class Thread with a little modification.
class ThreadingGraphAnalyzer(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

#Multiprocessing



def worker(graph, distance, node_dict,index_nodes,return_dict, processor):#single process function
    # calculate the subgraph with grater goodness/fairness in a slice set of the nodes
    # for preventing the fact that i will try to calculate the combination with the no connected
    # node. Otherwise we have 2^number of nodes combination!
    # param graph (directed networkx graph)
    # parm     index_nodes   (dict)   : dict key-value as node-fairness_value
    # parm        distance     (int)         : size of the subgraph (distance from node in bfs mode
    # param     processor               :the number of the processor
    # param     return_dict             :where i will put all the processors solution
    for elem in node_dict:
        nodes = nx.descendants_at_distance(graph, elem, distance)
        for nodes in combinations(nodes, distance):
            subgraph = graph.subgraph(nodes)

            # check if it is connected
            if nx.is_weakly_connected(subgraph):
                subgraph_goodness = 0
                for node in nodes:
                    # check if it has a goodness value, remember some value hasn't it
                    if node in index_nodes.keys():
                        subgraph_i = subgraph_goodness + index_nodes[node]
                # check if we need to update the values
                if subgraph_i > max_goodness:
                    max_goodness = subgraph_i
                    final_nodes_id = nodes
        return_dict[processor] = final_nodes_id












#Threading Old
def singleThreadPiece(graph, fairness_nodes,number):
        min_fairness = math.inf
        final_nodes_id = []
        start_time = time.monotonic()
        for node in combinations(graph, number):
            subgraph = graph.subgraph(node)

            # check if it is connected
            subgraph_fairness = subgraph_fairness + fairness_nodes[node]

                # check if we need to update the values
            if subgraph_fairness < min_fairness:
                min_fairness = subgraph_fairness
                final_nodes_id = node
        end_time = time.monotonic()
        print(end_time- start_time)#The time for computing the subgraph
        return final_nodes_id


























