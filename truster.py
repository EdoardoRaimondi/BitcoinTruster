import networkx as nx
import MyUtility
import numpy as np
import pandas as pd
from GraphAnalyzer import GraphAnalyzer

# This programs analyize Bitcoin trust network transaction. 
#
# The file columns format is:
# SOURCE TARGET RATE TIME
# where
# SOURCE: node id of source, i.e., rater
# TARGET: node id of target, i.e., ratee
# RATING: the source's rating for the target, ranging from -10 to +10 in steps of 1
# TIME: the time of the rating, measured as seconds since Epoch.

def main():  


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

    analyzer = GraphAnalyzer(graph)

    print("---STATISTICAL ANALYSIS---")
    print("Are transaction casual ? {}".format(analyzer.are_transations_casual(0.05)))

    # calculate degree of all nodes of the graph
    degree_nodes, max_degree_node = analyzer.graph_in_degree()

    # calculate the closeness centrality for all node in the graph
    centrality_nodes, max_centrality_node = analyzer.graph_centrality()

    # calculate the betweenness centrality for all node in the graph
    betweenness_nodes, max_betweennes_node = analyzer.graph_betweenness()

    # graph goodnees score
    print("goodness score of a random node : {}".format(analyzer.node_goodness(905)))
    goodness_nodes, max_goodness_node, min_goodness_node = analyzer.graph_goodness()

    # graph fariness score
    fairness_nodes, max_fairness_node, min_fairness_node = analyzer.graph_fairness()

    # first briefly analyses
    print("  NODE   |  DEGREE  | CLOSENESS | BETWEENNESS | GOODNESS |")
    print("   {}   | {} | {} |{}|{}".format(max_degree_node, degree_nodes[max_degree_node], centrality_nodes[max_degree_node], betweenness_nodes[max_degree_node], goodness_nodes[max_degree_node]))
    print("   {}   | {} | {} |{}|{}".format(max_centrality_node, degree_nodes[max_centrality_node], centrality_nodes[max_centrality_node], betweenness_nodes[max_centrality_node], goodness_nodes[max_centrality_node]))
    print("   {}   | {} | {} |{}|{}".format(max_betweennes_node, degree_nodes[max_betweennes_node], centrality_nodes[max_betweennes_node], betweenness_nodes[max_betweennes_node], goodness_nodes[max_betweennes_node]))
    print("   {}   | {} | {} |{}|{}".format(max_goodness_node, degree_nodes[max_goodness_node], centrality_nodes[max_goodness_node], betweenness_nodes[max_goodness_node], goodness_nodes[max_goodness_node]))
    print("   {}   | {} | {} |{}|{}".format(min_goodness_node, degree_nodes[min_goodness_node], centrality_nodes[min_goodness_node], betweenness_nodes[min_goodness_node], goodness_nodes[min_goodness_node]))

    # ----------------------------------------------------------
    #                     PRINT SOME GRAPHS
    # ----------------------------------------------------------

    # graph that show degree - closenness centrality and betweenness centrality
    MyUtility.drawGraph_Centrality(degree_nodes, centrality_nodes, betweenness_nodes, 100)

    # graph that show goodness-fairness for 100 nodes
    MyUtility.drawGraphGoodFair(goodness_nodes, fairness_nodes, 20)

    # ----------------------------------------------------------
    #                   PRINT SOME SUBGRAPHS
    # ----------------------------------------------------------

    # subgraph with the node that has max centrality
    #MyUtility.drawsubgraph(graph, max_centrality_node, 2) 

    # subgraph for goodness node
    nodes_id_goodness = analyzer.subgraphGoodness(goodness_nodes, 2)

    # subgraph for fairness node
    nodes_id_fairness = analyzer.subgraphFairness(fairness_nodes, 2)

if __name__ == "__main__":
    main()