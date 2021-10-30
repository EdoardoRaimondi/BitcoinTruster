import networkx as nx
import time
import matplotlib.pyplot as plt
from networkx.algorithms.centrality.betweenness import betweenness_centrality
from networkx.algorithms.centrality.closeness import closeness_centrality
from networkx.classes.function import is_directed, number_of_nodes, subgraph
import pandas as pd

# This programs analyize Bitcoin trust network transaction. 
#
# The file columns format is:
# SOURCE TARGET RATE TIME
# where
# SOURCE: node id of source, i.e., rater
# TARGET: node id of target, i.e., ratee
# RATING: the source's rating for the target, ranging from -10 to +10 in steps of 1
# TIME: the time of the rating, measured as seconds since Epoch.

# --------------------------------- 
#        FILE MANIPULATION
# ---------------------------------
# read csv file
df = pd.read_csv('soc-sign-bitcoinotc.csv')
# clean it (remove last column)
df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
columns_name = ['source', 'target', 'weight']
# add columns names 
df.columns = columns_name
# transform pandas dataframe into a directed graph
graph = nx.from_pandas_edgelist(df, 'source', 'target', ['weight'], create_using=nx.DiGraph())

# ---------------------------------
#             ANALYSES
# ---------------------------------

# print the node with the most closennes centrality score
# what does closeness centrality mean in our graph? Has it sense to consider it?
start_time = time.monotonic()
centrality_nodes = closeness_centrality(graph)
end_time = time.monotonic()
max_centrality_node = max(centrality_nodes, key = lambda x: centrality_nodes[x])
print("max closeness node, closeness: {}, {}".format(max_centrality_node, centrality_nodes[max_centrality_node])) 
print("execution time: {}".format(end_time-start_time)) #my pc 90sec

# print the node with th emost between centrality score
# is betweenness centrality interesting or not? What is it mean?
start_time = time.monotonic()
betweenness_nodes = betweenness_centrality(graph)
end_time = time.monotonic()
max_betweennes_node = max(betweenness_nodes, key = lambda x: betweenness_nodes[x])
print("max betweenness node, betweenness: {}, {}".format(max_betweennes_node, betweenness_nodes[max_betweennes_node]))  
print("execution time: {}".format(end_time-start_time)) #my pc 196sec

# calculate the degree for all nodes
# print out the node with greater degree and its value
start_time = time.monotonic()
degree_nodes = dict(graph.degree(graph.nodes)) # need a conversion tu use always the same lambda funciton
end_time = time.monotonic()
max_degree_node = max(degree_nodes, key = lambda x: degree_nodes[x])
print("node: {}, degree: {}".format(max_degree_node, degree_nodes[max_degree_node]))
print("execution time: {}".format(end_time-start_time)) #my pc 0.01sec

# firs biefly analyses
print("the node with greater centrality {}, has degree {} and betweenness {}".format(max_centrality_node, degree_nodes[max_centrality_node] ,betweenness_nodes[max_centrality_node]))
print("the node with greater betweenness {}, has degree {} and centrlity {}".format(max_betweennes_node, degree_nodes[max_betweennes_node] ,centrality_nodes[max_betweennes_node]))
print("the node with greater degree {}, has centrality {} and betweenness {}".format(max_degree_node, centrality_nodes[max_degree_node] ,betweenness_nodes[max_degree_node]))

# ----------------------------------------------------------
#                   PRINT SOME SUBGRAPHS
# ----------------------------------------------------------

# subgraph with the node that has max centrality
subnodes = graph.neighbors(max_centrality_node)
subgraph = graph.subgraph(subnodes)
figure1 = plt.subplot(121)
nx.draw(subgraph, with_labels=True)

# subgraph with the node that has max degree
subnodes = graph.neighbors(max_degree_node)
subgraph = graph.subgraph(subnodes)
figure1 = plt.subplot(121)
nx.draw(subgraph, with_labels=True)