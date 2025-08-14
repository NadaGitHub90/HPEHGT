import os
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import scipy.io as sio

from torch_geometric.data import HeteroData
from torch_geometric.utils import degree

from HPEHGT.hpeutils.data import get_dataset, train_val_test_split
from HPEHGT.hpeutils.utils import set_random_seed, adj_to_coo
from HPEHGT.hpeutils.draw import (
    draw_loss, draw_acc, visualize_embedding,
    draw_time_per_epoch, draw_cumulative_time
)
from HPEHGT.model.DBLP.hpehgt import HPEHGTModel
from sklearn.metrics import f1_score
from dataclasses import dataclass

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def load_args():
    parser = argparse.ArgumentParser(description='HPEHGT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=52, help='random seed')
    parser.add_argument('--dataset', type=str, default='DBLP', help='name of dataset')
    parser.add_argument('--hidden-dim', type=int, default=64, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.4, help="dropout-rate")
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--num-layers', type=int, default=2, help="number of transformer encoders")
    parser.add_argument('--heads', type=int, default=4, help="number of transformer multi-heads")
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--use_lr_schedule', type=bool, default=False, help='whether use warmup?')
  
    parser.add_argument('--convergence_epoch', type=int, default=50, help="number of epochs for warmup")
    parser.add_argument('--use_early_stopping', type=bool, default=True, help="whether to use early stopping")
    parser.add_argument('--use_pre_train_se', type=bool, default=False, help="whether to use pre-trained SE")
    parser.add_argument('--use_lgt', type=bool, default=False, help="whether to use LGT")
    parser.add_argument('--use_kd', type=bool, default=False, help="whether to use KD")
    parser.add_argument('--patience', type=int, default=200, help='early stopping patience')
    
    parser.add_argument('--lap-dim', type=int, default=16, help='Number of nontrivial Laplacian eigenvectors to compute')
   
    parser.add_argument('--gate-bias-meta', type=float, default=0.0,help='initial bias for MetaPE gating MLP')
    parser.add_argument('--gate-bias-lap',  type=float, default=0.0,help='initial bias for LapPE gating MLP')
    
    args = parser.parse_args()
    return args

def train(model, data, original_A, y, train_mask, deg, loss_fn, optimizer):
    model.train()
    loss_pe, out, embs, _, _ = model(data, original_A, deg)
    loss_acc = loss_fn(out[train_mask], y[train_mask])
    l = loss_pe + loss_acc

    #*****************************************
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    marco_f1 = f1_score(y_true=y[train_mask], y_pred=pred[train_mask], average='macro')
    mirco_f1 = f1_score(y_true=y[train_mask], y_pred=pred[train_mask], average='micro')

    return l.cpu().detach().numpy(), marco_f1, mirco_f1


def validate(model, data, original_A, y, val_mask, deg, loss_fn):
    model.eval()
    with torch.no_grad():
        loss_pe, out, embs, _, _ = model(data, original_A, deg)

        loss_acc = loss_fn(out[val_mask], y[val_mask])
        l = loss_pe + loss_acc
        pred = out.argmax(dim=1)
        marco_f1 = f1_score(y_true=y[val_mask], y_pred=pred[val_mask], average='macro')
        mirco_f1 = f1_score(y_true=y[val_mask], y_pred=pred[val_mask], average='micro')
    return l.cpu().detach().numpy(), marco_f1, mirco_f1


def test(model, data, original_A, y, test_mask, deg):
    model.eval()
    with torch.no_grad():
        loss_pe, out, embs, pe_Q, pe_K = model(data, original_A, deg)
        pred = out.argmax(dim=1)
        torch.save(pe_Q, 'DBLP_Q.pth')
        torch.save(pe_K, 'DBLP_K.pth')
        marco_f1 = f1_score(y_true=y[test_mask], y_pred=pred[test_mask], average='macro')
        mirco_f1 = f1_score(y_true=y[test_mask], y_pred=pred[test_mask], average='micro')
    return marco_f1, mirco_f1, embs




def main():
    global args
    args = load_args()
    print(args)

    # set_random_seed(args.seed)
    dataset = get_dataset(args.dataset, True)
    data = dataset[0]
    print(data)

    node_types, edge_types = data.metadata()
    num_nodes = (data[node_types[0]]['x']).shape[0]
    num_classes = len((data[node_types[0]]['y']).unique())
    num_edge_types = 3
    in_dim_a = data['author']['x'].shape[1]
    in_dim_p = data['paper']['x'].shape[1]
    in_dim_t = data['term']['x'].shape[1]

    # Initial the features of conference nodes to 0, since conference nodes have no features.
    data['conference']['x'] = torch.zeros((data['conference']['num_nodes'], args.hidden_dim))

    original_A = []

    edge_index_p_c = data[edge_types[3]]['edge_index']
    edge_index_p_t = data[edge_types[2]]['edge_index']
    edge_index_a_p = data[edge_types[0]]['edge_index']

    adj_p_c_sp = torch.sparse_coo_tensor(indices=edge_index_p_c, values=torch.ones(edge_index_p_c.shape[1]),
                                         size=(14328, 20))
    adj_p_c = adj_p_c_sp.to_dense()
    adj_c_p = adj_p_c.t()

    adj_p_t_sp = torch.sparse_coo_tensor(indices=edge_index_p_t, values=torch.ones(edge_index_p_t.shape[1]),
                                         size=(14328, 7723))
    adj_p_t = adj_p_t_sp.to_dense()
    adj_t_p = adj_p_t.t()

    adj_a_p_sp = torch.sparse_coo_tensor(indices=edge_index_a_p, values=torch.ones(edge_index_a_p.shape[1]),
                                         size=(4057, 14328))
    adj_a_p = adj_a_p_sp.to_dense()
    adj_p_a = adj_a_p.t()

    adj_pcp = torch.matmul(adj_p_c_sp, adj_c_p)
    adj_aca = torch.matmul(torch.matmul(adj_a_p_sp, adj_pcp), adj_p_a)
    adj_ptp = torch.matmul(adj_p_t_sp, adj_t_p)
    adj_ata = torch.matmul(torch.matmul(adj_a_p_sp, adj_ptp), adj_p_a)
    adj_apa = torch.matmul(adj_a_p_sp, adj_p_a)

    # Matepath-based adjacency matrices
    original_A.append(adj_apa)
    original_A.append(adj_ata)
    original_A.append(adj_aca)

    # Calculate degree
    all_A = original_A[0]
    for i in range(1, 3):
        all_A += original_A[i]
    deg = degree(index=all_A.nonzero().t()[0], num_nodes=num_nodes)
    deg = deg.to(device)
     

    original_A = torch.stack(original_A, dim=0)

    torch.save(original_A, 'DBLP_origin_A.pth')

    #********** added here for laplacian eigenvectors calculation from **********
    # ── Step 1: compute & save normalized‐Laplacian eigenvectors ──
    # sum over all meta‐path adjacencies → single N×N movie–movie graph
    A_sum = original_A.sum(dim=0).cpu().numpy()         # (N, N)
    # normalized Laplacian
    L = csgraph.laplacian(A_sum, normed=True)

    k = args.lap_dim
    # compute k+1 smallest; drop the trivial (constant) eigenvector at idx 0
    vals, vecs = eigsh(L, k=k+1, which='SM', tol=1e-3)
    # take the k non-trivial eigenvectors
    lap_evecs = vecs[:, 1:k+1]                         # (N, k)

    #******** we added this for Eigenvalue‐Informed Gating: grab the corresponding eigenvalues***************
    eigvals = vals[1:k+1]         # numpy array of shape (k,)
    #****************************************
    norms = np.linalg.norm(lap_evecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    lap_evecs = lap_evecs / norms                      # still numpy (N, k)

    lap_t = torch.from_numpy(lap_evecs).float().to(device)  # (num_nodes, lap_dim)
    # ── Eigenvalue-Informed Gating prep: grab the k nontrivial eigenvalues ──
    lap_eigvals_np = vals[1:k+1]
    lap_eigvals = torch.from_numpy(lap_eigvals_np).float().to(device)  # (lap_dim,)
    #*******************
    print(f"[DEBUG] Lap PE shape: {lap_t.shape}, mean ℓ₂ norm = "f"{lap_t.norm(p=2,dim=1).mean():.3f}", flush=True)
    np.save('author_lap_evecs.npy', lap_evecs)
    print(f"[LapPE] saved P_lap_evecs.npy shape={lap_evecs.shape}", flush=True)
    # ── also attach Lap‐PE to data for the model ──
    lap_tensor = torch.from_numpy(lap_evecs).to(device)  # (N, lap_dim)
    data['author'].lap = lap_tensor
    # ──************************ end LapPE block ──********************

    # degree + splits
    all_A = original_A.sum(dim=0)
    deg = degree(all_A.nonzero().t()[0], num_nodes=num_nodes).to(device)


    train_mask, val_mask, test_mask = train_val_test_split(num_nodes=num_nodes, y=data[node_types[0]]['y'], train_p=0.5,
                                                           val_p=0.25)

    data = data.to(device)

    original_A = original_A.to(device)
    y = data[node_types[0]]['y'].to(device)



    @dataclass
    class ACMconfig:
        
        num_nodes:      int
        in_dim_a:       int   # author
        in_dim_p:       int   # paper
        in_dim_t:       int   # term
        hidden_dim:     int
        svd_dim:        int
        classes:        int
        num_heads:      int
        dropout:        float
        bias:           bool
        num_blocks:     int
        num_metapaths:  int
        num_gnns:       int
        lap_dim:        int
        gate_bias_meta: float
        gate_bias_lap:  float
        lap_eigvals:    torch.Tensor



    cfg = ACMconfig(
        num_nodes      = num_nodes,
        in_dim_a       = data['author'].x.shape[1],
        in_dim_p       = data['paper'].x.shape[1],
        in_dim_t       = data['term'].x.shape[1],
        hidden_dim     = args.hidden_dim,
        svd_dim        = 16,
        classes        = num_classes,
        num_heads      = args.heads,
        dropout        = args.dropout,
        bias           = True,
        num_blocks     = args.num_layers,
        num_metapaths  = 3,
        num_gnns       = 3,
        lap_dim        = args.lap_dim,
        gate_bias_meta = args.gate_bias_meta,
        gate_bias_lap  = args.gate_bias_lap,
        lap_eigvals    = lap_eigvals,
    )


    model = HPEHGTModel(config=cfg, metadata=data.metadata()).to(device)








    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1, last_epoch=-1)

    loss_fn = nn.CrossEntropyLoss().to(device)

    save_path = 'HPE_models/' + args.dataset

    Train_Loss = []
    Train_macro_f1 = []
    Train_micro_f1 = []
    Val_Loss = []
    Val_macro_f1 = []
    Val_micro_f1 = []
    # ── timing buffers ──
    Train_time = []
    Val_time   = []


    patience = args.patience
    count = 0
    max_val_acc = 0
    for i in range(args.epochs):
        # ── train + time it ──
        t0 = time.time()
        train_loss, train_macro_f1, train_micro_f1 = train(model, data, original_A, y, train_mask, deg,
                                                           loss_fn,
                                                           optimizer)
        Train_time.append(time.time() - t0)

        Train_Loss.append(train_loss)



        Train_macro_f1.append(train_macro_f1)
        Train_micro_f1.append(train_micro_f1)
        # ── validate + time it ──
        v0 = time.time()
        val_loss, val_macro_f1, val_micro_f1 = validate(model, data, original_A, y, val_mask, deg, loss_fn)
        Val_time.append(time.time() - v0)
        Val_macro_f1.append(val_macro_f1)
        Val_micro_f1.append(val_micro_f1)
        Val_Loss.append(val_loss)
        test_macro_f1, test_micro_f1, _ = test(model, data, original_A, y, test_mask, deg)

        if i % 10 == 0:
            print(
                'Epoch {:03d}'.format(i),
                '|| train',
                'loss : {:.3f}'.format(float(train_loss)),
                ', macro_f1 : {:.2f}%'.format(train_macro_f1 * 100),
                ', micro_f1 : {:.2f}%'.format(train_micro_f1 * 100),
                '|| val',
                'loss : {:.3f}'.format(float(val_loss)),
                ', macro_f1 : {:.2f}%'.format(val_macro_f1 * 100),
                ', micro_f1 : {:.2f}%'.format(val_micro_f1 * 100),
                '|| test',
                ', macro_f1 : {:.2f}%'.format(test_macro_f1 * 100),
                ', micro_f1 : {:.2f}%'.format(test_micro_f1 * 100),
            )
        if i > args.convergence_epoch and args.use_lr_schedule:
            lr_scheduler.step()

        if args.use_early_stopping:
            if count <= patience:
                if max_val_acc >= Val_macro_f1[-1]:
                    if count == 0:
                        path = os.path.join(save_path, 'best_network_DBLP.pth')
                        torch.save(model.state_dict(), path)
                    count += 1
                else:
                    count = 0
                    max_val_acc = Val_macro_f1[-1]

    # ── final test + embedding plot ──
    _, _, embs = test(model, data, original_A, y, test_mask, deg)
    visualize_embedding(outputs=embs[test_mask].cpu(),labels=y[test_mask].cpu(),name=args.dataset + '_emb')

    draw_loss(Train_Loss, len(Train_Loss), args.dataset, 'Train')
    draw_acc(Train_macro_f1, len(Train_macro_f1), args.dataset, 'Train_macro_f1')
    draw_acc(Train_micro_f1, len(Train_micro_f1), args.dataset, 'Train_micro_f1')
    draw_loss(Val_Loss, len(Val_Loss), args.dataset, 'Val')
    draw_acc(Val_macro_f1, len(Val_macro_f1), args.dataset, 'Val_macro_f1')
    draw_acc(Val_micro_f1, len(Val_micro_f1), args.dataset, 'Val_micro_f1')
    # ── timing visualizations ──
    draw_time_per_epoch(Train_time, Val_time, args.dataset)
    draw_cumulative_time(Train_time, Val_time, args.dataset)
    # ── print final DBLP test metrics ──
    tm, tmi, _ = test(model, data, original_A, y, test_mask, deg)
    print(f"test_macro_f1:{tm*100:.2f}")
    print(f"test_micro_f1:{tmi*100:.2f}")


if __name__ == "__main__":
    main()
