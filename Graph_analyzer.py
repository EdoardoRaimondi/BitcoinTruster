
class Graph_analyzer:
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

        weights = []
        for n in self.graph.predecessors(node): # I want the entering nodes 
            weights.append(self.graph[n][node]["weight"])

        return sum(weights)/self.graph.in_degree(node)

    def graph_goodness(self, nodes_number):
        # Calculate goodnees of all the nodes of the graph
        # param graph (directed networkx graph) 
        # param nodes_number  (int)    : number of higher degree nodes to consider.
        #                                If no values is provided, the entire graph will be considered.
        # return dict node - goodness

        if(nodes_number == None):
            # consider the entire graph
            nodes_number = self.graph.number_of_nodes

        node_in_degree_dict = {}
        for node, degree in self.graph.in_degree(self.graph.nodes):
            # create a proper dictionary instead of a "DegreeView" as provided by in_degree networkx method
            node_in_degree_dict[node] = degree

        # now sort. Horrible, need to be refactored 
        sorted_node_degree_dict = {}
        sorted_degrees = sorted(node_in_degree_dict.values())
        for degree in sorted_degrees:
            for node in node_in_degree_dict.keys():
                if node_in_degree_dict[node] == degree:
                    sorted_node_degree_dict[node] = node_in_degree_dict[node]
                    break        

        # finally return the goodnees of our nodes
        nodes_goodnesses = {}
        for node in node_in_degree_dict.keys():
            nodes_number -= 1
            nodes_goodnesses[node] = self.node_goodness(node)
            if(nodes_number == 0): # I'm done
                return nodes_goodnesses




        