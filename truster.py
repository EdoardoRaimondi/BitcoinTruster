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
columns_name = ['source', 'target', 'rate']
# add columns names 
df.columns = columns_name
# transform pandas dataframe into a directed graph
graph = nx.from_pandas_edgelist(df, 'source', 'target', ['rate'], create_using=nx.DiGraph())

# ---------------------------------
#             ANALYSES
# ---------------------------------

# print the node with the most closennes centrality score
# what does closeness centrality mean in our graph? Has it sense to consider it?
start_time = time.monotonic()
closennes_nodes = closeness_centrality(graph)
end_time = time.monotonic()
max_centrality_node = max(closennes_nodes, key = lambda x: closennes_nodes[x])
print("max closeness node, closeness: {}, {}".format(max_centrality_node, closennes_nodes[max_centrality_node])) #905  
print("execution time: {}".format(end_time-start_time)) #my pc 90sec

# ----------------------------------------------------------
#        PRINT A SUBGRAPH WITH GREATER CENTRALITY
# ----------------------------------------------------------
subnodes = graph.neighbors(max_centrality_node)
subgraph = graph.subgraph(subnodes)
figure1 = plt.subplot(121)
nx.draw(subgraph, with_labels=True)