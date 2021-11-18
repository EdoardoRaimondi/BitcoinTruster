# BitcoinTruster

**TITLE**

Analysis of bitcoin transactions trustability graph

**MOTIVATIONS**

Since OTC transactions are anonymously performed, our aim is to investigate the transactions graph and give to it a proper formalism and interpretation, in order to enlarge its expression power. Since the transactions are a bipartite accord we are interest to analyze which node has an high good reputation and which not. \\

**Dataset**

We use a public dataset given by Standford University at the following link: <https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html> .
It is composed by: \\
    1. Nodes: 5881 \\
    2. Edges: 35592\\
    3. Range of edge weight: -10 to +10\\

**METHODS**

- node level : goodness and degree of nodes. Investigation about the relationship between goodness of a node and the number of its transactions. Moreover, we analyze also the fairness of a node that it is the mean of the variation w.r.t. the vote that that node gives to another one. Furthermore, we do a comparison between the two metrics defined in order to find the best node in both case. \\
- graph level : search for particular subgraphs (made of 2, 3 or 4 nodes) that must have good reputation (high goodness and low fairness). Indeed if we are a new node, we want contact them in order to have a satisfactory transation. (after that we can calculate the probability to contacts these node or to stay in that subgraph [if time we can also do it]) eventually recalibration of the weights concordly) \\

**EXPERIMENTS**

- using Networkx we analyze closeness centrality, betweenness centrality and degree of each node. We do it in order to understand the network structure, since it can be unbalanced and affects our metrics.
- we give an interpretation of goodness for a node, that it is related to the evaluation that a node received and to the input degree.
- we give an interpretation of fairness for a node, that it is related to the vote that a node gives to another and the variation of it with respect the mean goodness for that node.
- we do a comparison between the two metrics. In this way we can understand which node receives a good evaluations and provide fairly ones.
- finally we search in the network a subgraph (made of 2 or 3 or 4 node) that has the highest goodness and the lowest fariness. If a new user is interested in doing a transaction in this graph, it could want to 'keep in touch' with such nodes.
- for statistical purposes, we aim to assert whether the transactions between users (so if A is connected rather than B) in our graph are casual or not ( hypotesis testing Erdos-Renyi-Gilbert random graph generation )

**Related Works**

    1. S. Kumar, F. Spezzano, V.S. Subrahmanian, C. Faloutsos. Edge Weight Prediction in Weighted Signed Networks.
    2. S. Kumar, B. Hooi, D. Makhija, M. Kumar, V.S. Subrahmanian, C. Faloutsos. REV2: Fraudulent User Prediction in Rating Platforms.

**Machine used**

● cpu: AMD Ryzen 9 3950X 16-Core Processor 3.49 GHz
● ram: 16 GB
● video card: Nvidia gefoce 2080 ti oc edition with 11GB of
GDDR6 vram 1350 MHz boostable to 1665 mhz with 4351
CUDA Cores
● Memory: 1.5 TB of SSD and 1 TB of HDD

**AUTHORS**

Edoardo Raimondi
Enrico Sabbatini
Paolo Forin
