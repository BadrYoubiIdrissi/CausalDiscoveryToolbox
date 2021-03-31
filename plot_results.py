import seaborn as sns
import networkx as nx
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve
from cdt.data.acyclic_graph_generator import scale
from tqdm import tqdm

sns.set_theme("poster", style="white")

# run_root = "multirun/2021-03-31/09-34-10"
# run_root = "multirun/2021-03-30/18-30-03"
run_root = "multirun/2021-03-31/09-39-36"
run_root = "multirun/2021-03-31/10-16-07"
run_root = "multirun/2021-03-31/10-23-20"
# run_root = "multirun/2021-03-31/10-34-55"
run_root = "multirun/2021-03-31/10-57-43"
run_path = os.path.join(run_root, '1')
output_path = os.path.join(run_path, "figures")
if not os.path.exists(output_path):
    os.mkdir(output_path)

original_data = pd.read_csv(os.path.join(run_path, "data.csv"), index_col=0)
predicted_graphs = []
for i in tqdm(range(16)):
    predicted_graphs.append(np.load(os.path.join(run_root, str(i),"predicted_graph.npy")))
predicted_graph = np.concatenate(predicted_graphs)
real_graph = nx.read_adjlist(os.path.join(run_path, "true_graph"), create_using=nx.DiGraph)
print(np.mean(predicted_graph), np.max(predicted_graph), np.min(predicted_graph))
# for i in tqdm(range(5)):
#     original_data = pd.read_csv(os.path.join(run_root, str(i), "data.csv"), index_col=0)
#     predicted_data = np.load(os.path.join(run_root, str(i),"data_output.npy"))
#     predicted_data = pd.DataFrame(scale(predicted_data[0]), columns=original_data.columns)
#     ori = pd.DataFrame(scale(original_data.copy()), columns=original_data.columns)
#     predicted_data["type"] = "predicted"
#     ori["type"] = "true"
#     all_data = pd.concat([ori, predicted_data])
#     sns.pairplot(all_data, hue="type", plot_kws=dict(marker="+", linewidth=0.5))
#     plt.savefig(os.path.join(run_root, str(i),"dataset_comp.png"))

plt.figure()
real_graph_adj = np.array(nx.adjacency_matrix(real_graph, real_graph.nodes(), weight=None).todense())
prec, rec, thresh = precision_recall_curve(real_graph_adj.ravel(), predicted_graph.mean(axis=0).ravel())
denom = (prec+rec)
denom[denom==0] = 1
f1score = 2*prec*rec/denom
print(max(f1score), thresh[np.argmax(f1score)])
thres = 0.7
aucscore = auc(rec, prec)

plt.figure(figsize=(14,10))
no_skill = (real_graph_adj).sum()/real_graph_adj.size
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(rec, prec, marker='.', label='Model')
# axis labels
plt.title(f"AUC={aucscore:.2e}")
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.savefig(os.path.join(output_path,"rec_prec.png"))

plt.figure(figsize=(10,10))
g = predicted_graph.mean(axis=0)
print(g)
predicted_graph_th = (g>thres)*((g>g.T+0.01)).astype(int)
# diff = (predicted_graph_th == real_graph_adj)*predicted_graph_th + 2*(real_graph_adj*(1-predicted_graph_th)) + 3*(predicted_graph_th*(1-real_graph_adj))
# predicted_graph_nx = nx.relabel_nodes(nx.DiGraph(diff),
#                                         {idx: i for idx,
#                                             i in enumerate(original_data.columns)})
# nx.draw_networkx(output, font_size=8)
# plt.savefig("predicted_network.png")
# plt.figure()
# nx.draw_networkx(graph)
# plt.savefig("real_network.png")
from matplotlib.colors import ListedColormap

edge_cmap = ListedColormap([np.array([0,1,0]), np.array([0,0,1]), np.array([1,0,0])])


nx.draw_networkx(nx.relabel_nodes(nx.DiGraph(real_graph_adj*(1-predicted_graph_th)),{idx: i for idx,i in enumerate(original_data.columns)}), pos=nx.circular_layout(real_graph), node_size=1000, font_size=15, font_color="white", edge_color="blue", arrows=True)
nx.draw_networkx(nx.relabel_nodes(nx.DiGraph(predicted_graph_th*(1-real_graph_adj)),{idx: i for idx,i in enumerate(original_data.columns)}), pos=nx.circular_layout(real_graph), node_size=1000, font_size=15, font_color="white", edge_color="red", arrows=True)
nx.draw_networkx(nx.relabel_nodes(nx.DiGraph(real_graph_adj*predicted_graph_th),{idx: i for idx,i in enumerate(original_data.columns)}), pos=nx.circular_layout(real_graph), node_size=1000, font_size=15, font_color="white", edge_color="green", arrows=True)
plt.savefig(os.path.join(output_path,"predic_graph.png"))


