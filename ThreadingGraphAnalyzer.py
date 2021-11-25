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
def worker(procnum, return_dict, graph, fairness_nodes,slice_graph):
    """worker function"""
    print(str(procnum) + " present!")
    min_fairness = math.inf
    final_nodes_id = []
    start_time = time.monotonic()
    print("UUUUU")
    for nodes in slice_graph:
        subgraph = graph.subgraph(nodes)
        # check if it is connected
        if nx.is_weakly_connected(subgraph):
            subgraph_fairness = 0
            if nodes in fairness_nodes.keys():
                subgraph_fairness = subgraph_fairness + fairness_nodes[nodes]
            # check if we need to update the values
            if subgraph_fairness < min_fairness:
                min_fairness = subgraph_fairness
                final_nodes_id = nodes
    end_time = time.monotonic()
    print(end_time- start_time)#The time for computing the subgraph
    print("Final node!!!!!!")
    print(final_nodes_id)
    return_dict[min_fairness] = final_nodes_id



#Threading
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


