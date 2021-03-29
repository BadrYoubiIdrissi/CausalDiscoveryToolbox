import networkx as nx
from cdt.causality.graph import SAM
import matplotlib.pyplot as plt
import hydra
import numpy as np

@hydra.main(config_name="config")
def train(cfg):
    dataset = hydra.utils.instantiate(cfg.data)
    # data, graph = AcyclicGraphGenerator("polynomial", npoints=2000, nodes=5)
    data, graph = dataset.generate()
    obj = SAM(**cfg.model)
    #The predict() method works without a graph, or with a
    #directed or undirected graph provided as an input
    output = obj.predict(data)    #No graph provided as an argument
    # output = obj.predict(data, nx.Graph(graph))  #With an undirected graph
    # output = obj.predict(data, graph)  #With a directed graph
    #To view the graph created, run the below commands:
    nx.write_adjlist(output, "output_graph")
    nx.draw_networkx(output, font_size=8)
    plt.savefig("predicted_network.png")
    plt.figure()
    nx.draw_networkx(graph)
    plt.savefig("real_network.png")
if __name__=="__main__":
    train()