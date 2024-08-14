import networkx as nx
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.io import loadmat

def dimacs_reader(filename):
    G = nx.Graph()
    with open(filename, 'r+') as f:
        for line in f:
            line_read = line.strip() #removes the \n at the end
            words = line_read.split(' ')
            if words[0] == 'p':
                n_nodes = int(words[2])
            if words[0] == 'e':
                G.add_edge(int(words[1])-1, int(words[2])-1)
            #print(line_read)
    Adj = nx.to_numpy_array(G)
    return G, Adj, n_nodes

if __name__ == "__main__":
    #test_folder = "./"
    #graph_file = test_folder + sys.argv[1]
    graph_file = 'myciel3.col'
    G, Adj, n_nodes = dimacs_reader(graph_file)
    nx.draw(G, pos=nx.circular_layout(G), with_labels=True)
    plt.show()
    test_folder = "./test_graphs/"
    graph = test_folder + 'myciel3.mat'
    graph_dict = loadmat(graph)
    Adj_mat = graph_dict['M']
    G_m = nx.Graph(Adj_mat)
    nx.draw(G_m, pos=nx.circular_layout(G_m), with_labels=True)
    plt.show()
    print('Are .col and .mat graphs isomorphic? ', nx.is_isomorphic(G, G_m))