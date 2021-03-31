import hydra
import contextlib
import numpy as np

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

@hydra.main(config_name="config")
def train(cfg):
    from cdt.causality.graph import SAM
    import networkx as nx
    import numpy as np

    with temp_seed(42):
        dataset = hydra.utils.instantiate(cfg.data)
        dataset.init_variables()
        
    data, graph = dataset.generate()
        # data, graph = hydra.utils.instantiate(cfg.data)
    nx.write_adjlist(graph, "true_graph")
    data.to_csv("data.csv")
    obj = SAM(**cfg.model)
    graph_output, data_output = obj.predict(data)    #No graph provided as an argument

    np.save("predicted_graph", graph_output)
    np.save("data_output", data_output)
if __name__=="__main__":
    train()