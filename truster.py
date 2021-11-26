import networkx as nx
from networkx.classes.graph import Graph
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

    # print two graph to understand what we have in term of edges and degree
#    MyUtility.draw_histogram(graph,0)
#    MyUtility.draw_histogram(graph,1)
#    MyUtility.draw_histogram(graph,2)

    analyzer = GraphAnalyzer(graph)

    print("---STATISTICAL ANALYSIS---")
#    print("Are transaction casual ? {}".format(analyzer.are_transations_casual(0.05)))

    print("----DEGREE ANALYSIS----")
    # calculate degree of all nodes of the graph
    in_degree_nodes, max_degree_node = analyzer.graph_in_degree()
    out_degree_nodes = analyzer.graph_out_degree()

    # graph goodnees score
    print("goodness score of a random node : {}".format(analyzer.node_goodness(905, in_degree_nodes[max_degree_node])))
    goodness_nodes = analyzer.graph_goodness(max_degree = in_degree_nodes[max_degree_node])

    # graph fariness score
    fairness_nodes = analyzer.graph_fairness()

    # ----------------------------------------------------------
    #                     PRINT SOME GRAPHS
    # ----------------------------------------------------------

    # graph that show goodness-fairness for 100 nodes
#    MyUtility.draw_graph_good_fair(goodness_nodes, fairness_nodes, 20)

    # graph degree - goodness
#    MyUtility.draw_graph_scatter(list(graph.nodes()), in_degree_nodes, goodness_nodes, 'degree', 'goodness', 'in degree-goodness')

    # graph degree - fairness
#    MyUtility.draw_graph_scatter(list(graph.nodes()), out_degree_nodes, fairness_nodes, 'degree', 'fairness', 'out degree-fairness')

    # ----------------------------------------------------------
    #                   PRINT SOME SUBGRAPHS
    # ----------------------------------------------------------

    # subgraph for goodness node
    print("---SEARCH SUBGRAPHS---")
#    nodes_id_goodness = analyzer.search_subgraph(goodness_nodes, 2, 1) -> Dovrebbe funzionare da utilizzare al posto 
#                                                                          di subgraph_goodness e fairness con il type 1 o 2
#    nodes_id_goodness = analyzer.subgraph_goodness(goodness_nodes, 2) # 1, 1201
#    MyUtility.draw_subgraph(graph, list(nodes_id_goodness), goodness_nodes)

    # subgraph for fairness node
#    nodes_id_fairness = analyzer.subgraph_fairness(fairness_nodes, 2) # 695, 696
#    MyUtility.draw_subgraph(graph, list(nodes_id_fairness), fairness_nodes)

    # ----------------------------------------------------------
    #                    CALCULATE FEATURES
    # ----------------------------------------------------------

    print("---CLUSTERING---")
    # there is another assumptions:
    #   - we consider only the nodes that has fairness and goodness values 
#    nodes_features = MyUtility.give_node_features(list(graph.nodes()), goodness_nodes, fairness_nodes)
#    analyzer.cluster(2,nodes_features)

    # last, we want to rank each node using the goodness and fairness values
    ranking_nodes = MyUtility.ranking(graph.nodes(), goodness_nodes, fairness_nodes, 0.85)
    for i in range(0, 10):
        print("best node: {}, value: {}, goodness: {}, fairness: {}".format(list(ranking_nodes.keys())[i], 
        list(ranking_nodes.values())[i], goodness_nodes[list(ranking_nodes.keys())[i]], 
        fairness_nodes[list(ranking_nodes.keys())[i]]))

if __name__ == "__main__":
    main()