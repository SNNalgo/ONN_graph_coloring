import numpy as np
import GraphColoringSoln as GC
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import Monitor
import matplotlib.pyplot as plt
import sys

from scipy.io import loadmat

#np.random.seed(0)

def gen_permutation_matrix(phi):
    n = phi.shape[0]
    order = np.argsort(phi)
    P = np.zeros((n, n))
    for i in range(n):
        P[i,order[i]] = 1
    return P, order

def get_chr(P, A, order):
    n = A.shape[0]
    PAPT = P @ A @ P.T
    block_list = []
    color_blocks = []
    i = 0
    while (i<n):
        condition = 0
        block = 0
        color_block = []
        while (condition==0 and (i+block)<n):
            #if PAPT[i:i+block+1,i:i+block+1] == np.zeros((block+1,block+1)):
            if np.sum(np.abs(PAPT[i:i+block+1,i:i+block+1])) == 0:
                color_block.append(order[i+block])
                block = block+1
            else:
                condition = 1
        block_list.append(block)
        color_blocks.append(color_block)
        i = i + block
    return len(block_list), color_blocks

def order_based_chr(P, A, order):
    n = A.shape[0]
    P_shift = np.zeros_like(P)
    min_chr = n
    min_color_blocks = []
    for shift in range(n):
        for i in range(n):
            nin = (i+shift) % n
            P_shift[i,:] = P[nin,:]
        n_chr, color_blocks = get_chr(P_shift, A, order)
        if n_chr<min_chr:
            min_chr = n_chr
            min_color_blocks = color_blocks
    return min_chr, min_color_blocks

def compute_loss(phi, B, C):
    steps = phi.shape[0]
    Closs = np.zeros(steps)
    Bloss = np.zeros(steps)
    for n in range(steps):
        #loss[n] = np.sum((R + C - B)*(1 - (np.outer(np.cos(phi[n,:]), np.cos(phi[n,:])) - np.outer(np.sin(phi[n,:]), np.sin(phi[n,:])))))
        Closs[n] = np.sum((C)*(1 - (np.outer(np.cos(phi[n,:]), np.cos(phi[n,:])) + np.outer(np.sin(phi[n,:]), np.sin(phi[n,:])))))
        Bloss[n] = np.sum((-B)*(1 - (np.outer(np.cos(phi[n,:]), np.cos(phi[n,:])) + np.outer(np.sin(phi[n,:]), np.sin(phi[n,:])))))
        
    return Bloss, Closs

def gen_cyclic_graph(n):
    B = np.zeros((n, n))
    for i in range(n):
        B[i,i-1] = 1.0
        B[i,(i+1)%n] = 1.0
    return B

if __name__ == "__main__":
    
    #Test benchmark graphs
    
    test_folder = "./test_graphs/"
    graph = test_folder + sys.argv[1] + '.mat'
    lr = float(sys.argv[2])
    graph_dict = loadmat(graph)
    Adj = np.float64(graph_dict['M'])
    n_nodes = Adj.shape[0]
    
    #Test cyclic connected graphs
    '''
    n_nodes = int(sys.argv[1])
    Adj = gen_cyclic_graph(n_nodes)
    lr = 0.001
    '''
    #Test Peterson graph
    '''
    n_nodes = 10
    Adj = np.zeros((n_nodes, n_nodes))
    lr = 0.001
    
    Adj[0,2] = 1
    Adj[0,3] = 1
    Adj[0,5] = 1
    
    Adj[1,3] = 1
    Adj[1,4] = 1
    Adj[1,6] = 1
    
    Adj[2,0] = 1
    Adj[2,4] = 1
    Adj[2,7] = 1
    
    Adj[3,0] = 1
    Adj[3,1] = 1
    Adj[3,8] = 1
    
    Adj[4,1] = 1
    Adj[4,2] = 1
    Adj[4,9] = 1
    
    Adj[5,0] = 1
    Adj[5,6] = 1
    Adj[5,9] = 1
    
    Adj[6,1] = 1
    Adj[6,5] = 1
    Adj[6,7] = 1
    
    Adj[7,2] = 1
    Adj[7,6] = 1
    Adj[7,8] = 1
    
    Adj[8,3] = 1
    Adj[8,7] = 1
    Adj[8,9] = 1
    
    Adj[9,4] = 1
    Adj[9,5] = 1
    Adj[9,8] = 1
    '''
    B = Adj
    #print("B: ", B)
    conns = np.sum(B, axis=1)
    Cscale = 0.1
    C = (1.0 - Adj)
    connsC = np.sum(C, axis=1)
    
    init_range = 0.2*np.pi
    phi_init = np.random.uniform(-init_range, init_range, size=(n_nodes,))
    cos_phi_init = np.cos(phi_init)
    sin_phi_init = np.sin(phi_init)
    
    lrc = lr*(n_nodes - conns)/(n_nodes)
    lrr = Cscale*lr*conns/(n_nodes)
    print("min lrc: ", np.min(lrc))
    print("min lrr: ", np.min(lrr))
    #lrc = lr*(1/conns)
    #lrr = lr*(1/(n_nodes - conns))
    #lrc = lr/n_nodes
    #lrr = 0
    tau = 500
    decay = 1 - 1/tau
    sigma = 0.1
    #sigma = 0.0
    num_steps = 2000
    
    #nodes = GC.OScillatoryNeuron(shape=(n_nodes,), phi=phi_init, sigma=sigma, decay=decay, conns=conns, lr=lr, n_nodes=n_nodes)
    nodes = GC.OScillatoryNeuron(shape=(n_nodes,), phi=phi_init, sigma=sigma, decay=decay, lrc=lrc, lrr=lrr)
    connections = GC.GraphColorUpdate(shape=(n_nodes,), shape_mat=(n_nodes, n_nodes), B=B, C=C)
    phi_monitor = Monitor()
    
    nodes.cos_out.connect(connections.cos_in)
    nodes.sin_out.connect(connections.sin_in)
    
    connections.B_cos_out.connect(nodes.B_cos_in)
    connections.B_sin_out.connect(nodes.B_sin_in)
    connections.C_cos_out.connect(nodes.C_cos_in)
    connections.C_sin_out.connect(nodes.C_sin_in)
    phi_monitor.probe(nodes.phi, num_steps)
    
    nodes.run(condition=RunSteps(num_steps=num_steps), run_cfg=Loihi1SimCfg())
    phi_vals = phi_monitor.get_data()
    nodes.stop()
    phi_history = phi_vals['Process_0']['phi']
    
    last_phi = phi_history[-1,:]
    P, order = gen_permutation_matrix(last_phi)
    chromatic_number, min_color_blocks = order_based_chr(P, Adj, order)
    print("order-based chromatic number at simulation end: ", chromatic_number)
    print("color blocks: ", min_color_blocks)
    
    print_fr = 500
    chr_ns = []
    min_chr_n = n_nodes
    min_ind = 0
    for i in range(phi_history.shape[0]):
        if i%print_fr == print_fr-1:
            print("computing for iter: ", i+1)
            print("minimum chromatic number: ", min_chr_n)
        curr_phi = phi_history[i,:]
        P, order = gen_permutation_matrix(curr_phi)
        chr_n, _ = order_based_chr(P, Adj, order)
        if chr_n<min_chr_n:
            min_chr_n = chr_n
            min_ind = i
        chr_ns.append(chr_n)
    
    print("minimum order-based chromatic number during simulation: ", min_chr_n)
    Bloss, Closs = compute_loss(phi_history, B, C)
    
    phi_legend = []
    
    plt.figure(1)
    for i in range(n_nodes):
        plt.plot(phi_history[:,i])
        phi_legend.append("node" + str(i))
    plt.legend(phi_legend)
    plt.xlabel('steps')
    plt.ylabel('phi')
    plt.savefig('phi'+str(n_nodes)+'.png', dpi=300)
    plt.show()
    
    
    plt.figure(3)
    plt.plot(chr_ns)
    plt.xlabel('steps')
    plt.ylabel('chromatic number')
    plt.show()
    
    plt.figure(2)
    plt.plot(Bloss)
    plt.plot(Closs)
    plt.plot(Bloss + Closs)
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.legend(['cost loss', 'reward loss ', 'total loss'])
    plt.show()
