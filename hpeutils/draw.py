import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from torch_geometric.utils import degree, add_self_loops, to_scipy_sparse_matrix, is_undirected, \
    to_undirected, to_networkx
import networkx as nx
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['svg.fonttype'] = 'none'


def visualize_embedding(outputs, labels, name):
    outputs = outputs.detach().numpy()
    labels = labels.detach().numpy()

    model = TSNE(n_components=2)  
    node_pos = model.fit_transform(outputs)

    color_idx = {}
    for i in range(outputs.shape[0]):
        label = labels[i]
        color_idx.setdefault(label.item(), [])
        color_idx[label.item()].append(i)

    reorder_color_idx = {}
    for j in range(len(color_idx)):
        reorder_color_idx[j] = color_idx[j]

    for c, idx in reorder_color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.savefig('./HPE_figs/' + name + '_emb.svg', bbox_inches='tight')


def draw_heatmap(matrix, name, figsize):
    
    dict_ = {'label': 'Edge weight'}

    fig, ax = plt.subplots(figsize=figsize)
    node_idx = np.array(list(range(matrix.shape[0])))

    sns.heatmap(pd.DataFrame(matrix, columns=node_idx,
                             index=node_idx), annot=False, vmax=matrix.max(), vmin=matrix.min(), xticklabels=True,
                yticklabels=True, square=True, cmap="YlGnBu", cbar_kws=dict_)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    
    
    plt.savefig('./HPE_figs/' + name + '_reconstruct.svg', bbox_inches='tight')


def draw_loss(Loss_list, epochs, dataset, Type_name='Train'):
    plt.cla()
    x1 = range(1, epochs + 1)
    y1 = Loss_list
    plt.title(Type_name + ' loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epochs', fontsize=20)
    plt.ylabel(Type_name + ' loss', fontsize=20)
    plt.grid()
    plt.savefig('./HPE_figs/' + dataset + '/' + Type_name + '_loss.png')
    plt.show()


def draw_acc(acc_list, epochs, dataset, Type_name='Train'):
    plt.cla()
    x1 = range(1, epochs + 1)
    y1 = acc_list
    plt.title(Type_name + ' accuracy vs. epoch', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel(Type_name + ' accuracy', fontsize=20)
    plt.grid()
    plt.savefig('./HPE_figs/' + dataset + '/' + Type_name + '_accuracy.png')
    plt.show()




#    ******** add draw_time_per_epoch AND draw_cumulative_time functions ********* #

def draw_time_per_epoch(train_times, val_times, dataset):
    """Plot train & validation time (s) vs. epoch."""
    x = range(1, len(train_times) + 1)
    plt.cla()
    plt.plot(x, train_times, '.-', label='Train time/epoch')
    plt.plot(x, val_times,   '.-', label='Val time/epoch')
    plt.title('Time per epoch', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('time (s)', fontsize=20)
    plt.legend()
    plt.grid()
    os.makedirs(f'./HPE_figs/{dataset}', exist_ok=True)
    plt.savefig(f'./HPE_figs/{dataset}/time_per_epoch.png')
    plt.show()

def draw_cumulative_time(train_times, val_times, dataset):
    """Plot cumulative train & validation time (s) vs. epoch."""
    cum_train = np.cumsum(train_times)
    cum_val   = np.cumsum(val_times)
    x = range(1, len(train_times) + 1)
    plt.cla()
    plt.plot(x, cum_train, '.-', label='Cumulative train time')
    plt.plot(x, cum_val,   '.-', label='Cumulative val time')
    plt.title('Cumulative time vs. epoch', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('cumulative time (s)', fontsize=20)
    plt.legend()
    plt.grid()
    os.makedirs(f'./HPE_figs/{dataset}', exist_ok=True)
    plt.savefig(f'./HPE_figs/{dataset}/cumulative_time.png')
    plt.show()




