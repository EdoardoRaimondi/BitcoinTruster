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
def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def incrementingPlusOne(listA, limit4cell):
    listA[0] = listA[0]+1
    if(listA[0]>limit4cell-2):
        listA[0] = listA[0]-1
        for i in range(1,len(listA)):
            listA[i]= listA[i]+1
            if listA[i]> limit4cell-2:
                listA[i]=listA[i]-1
            else:
                break
    return


def limitForCountingChuncks(lower, number_cell, num_item):
    limit = list()
    for i in range(number_cell):
        limit.append(0)
    for i in range(lower):
        incrementingPlusOne(limit, num_item)
    return limit

def checkLimit(listA, limit):
    cnt = len(listA)-1
    while(listA[cnt]==0) and (cnt >0):
        if listA[cnt]>(limit-2):
            return False
        cnt = cnt - 1
    return True



def worker(processor,num_processors, return_dict, graph, fairness_nodes, node_dict, lower_limit, upper_limit):
    """worker function for processing"""
    min_fairness = math.inf
    num_nodes = len(node_dict)
    final_nodes_id = []
    nodes = list()
    start_time = time.monotonic()
    counting = limitForCountingChuncks(lower_limit, num_processors,len(node_dict))
    while(checkLimit(counting, upper_limit)):
        incrementingPlusOne(counting,num_nodes)
        for index in counting:
            nodes.append(node_dict[index])
        subgraph = graph.subgraph(nodes)
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
    print("processor: {}".format(processor))
    print("time: {}".format(end_time-start_time))
    return_dict[processor] = final_nodes_id



#Threading work in progress
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


