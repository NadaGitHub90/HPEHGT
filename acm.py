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

from HPEHGT.hpeutils.data import train_val_test_split
from HPEHGT.hpeutils.utils import set_random_seed, adj_to_coo
from HPEHGT.hpeutils.draw import (
    draw_loss, draw_acc, visualize_embedding,
    draw_time_per_epoch, draw_cumulative_time
)
from HPEHGT.model.ACM.hpehgt import HPEHGTModel
from sklearn.metrics import f1_score
from dataclasses import dataclass


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_args():
    parser = argparse.ArgumentParser(description='HPEHGT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=52, help='random seed')
    parser.add_argument('--dataset', type=str, default='ACM', help='name of dataset')
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
        torch.save(pe_Q, 'ACM_Q.pth')
        torch.save(pe_K, 'ACM_K.pth')
        marco_f1 = f1_score(y_true=y[test_mask], y_pred=pred[test_mask], average='macro')
        mirco_f1 = f1_score(y_true=y[test_mask], y_pred=pred[test_mask], average='micro')
    return marco_f1, mirco_f1, embs


def main():
    global args
    args = load_args()
    print(args)

    set_random_seed(args.seed)
    dataset = sio.loadmat('dataset/ACM3025.mat')
    data = HeteroData()
    print(f"→ Loaded full dataset: {data}")

    # Metapath edges
    #adj_ptp = torch.tensor(dataset['PTP'], dtype=torch.float32)
    adj_plp = torch.tensor(dataset['PLP'], dtype=torch.float32)
    adj_pap = torch.tensor(dataset['PAP'], dtype=torch.float32)


    # metapath edges for ACM
    data[('P','PAP','P')].edge_index = adj_to_coo(adj_pap)._indices()
    data[('P','PLP','P')].edge_index = adj_to_coo(adj_plp)._indices()
    #data[('P','PTP','P')].edge_index = adj_to_coo(adj_ptp)._indices()


    data['P'].x = torch.tensor(dataset['feature'], dtype=torch.float32)
    data['P'].y = torch.tensor(dataset['label'], dtype=torch.float32).nonzero()[:, 1]
    data['P'].train_idx = torch.tensor(dataset['train_idx'][0])
    data['P'].val_idx = torch.tensor(dataset['val_idx'][0])
    data['P'].test_idx = torch.tensor(dataset['test_idx'][0])
    print(data)
    
    node_types, edge_types = data.metadata()
    num_nodes = adj_pap.shape[0]
    num_classes = len((data['P'].y).unique())
    in_dim = data['P'].x.shape[1]

    original_A = []
    original_A.append(adj_pap)
    original_A.append(adj_plp)
    

    # Calculate degree
    all_A = original_A[0]
    for i in range(1, 2):
        all_A += original_A[i]
    deg = degree(index=all_A.nonzero().t()[0], num_nodes=num_nodes)
    deg = deg.to(device)

    data = data.to(device)
    original_A = torch.cat(original_A, dim=0).view(-1, num_nodes, num_nodes)
    original_A = original_A.to(device)

    A_sum = original_A.sum(dim=0).cpu().numpy()         # (N, N)
    L = csgraph.laplacian(A_sum, normed=True)

    k = args.lap_dim
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
    np.save('movie_lap_evecs.npy', lap_evecs)
    print(f"[LapPE] saved P_lap_evecs.npy shape={lap_evecs.shape}", flush=True)
    #attach Lap‐PE to data for the model ──
    lap_tensor = torch.from_numpy(lap_evecs).to(device)  # (N, lap_dim)
    data['P'].lap = lap_tensor
    # ──************************ end LapPE block ──********************
    all_A = original_A.sum(dim=0)
    deg = degree(all_A.nonzero().t()[0], num_nodes=num_nodes).to(device)
    train_mask, val_mask, test_mask = train_val_test_split(
        num_nodes=num_nodes,
        y=data['P'].y,
        train_p=0.5,
        val_p=0.25
    )

    data = data.to(device)
    original_A = original_A.to(device)
    y = data['P'].y.to(device)


    @dataclass
    class ACMconfig:
        num_nodes:      int
        x_input_dim:    int
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
        x_input_dim    = data['P'].x.shape[1],
        hidden_dim     = args.hidden_dim,
        svd_dim        = 16,
        classes        = num_classes,
        num_heads      = args.heads,
        dropout        = args.dropout,
        bias           = True,
        num_blocks     = args.num_layers,
        num_metapaths  = 2,
        num_gnns       = 3,
        lap_dim        = args.lap_dim,
        gate_bias_meta = args.gate_bias_meta,
        gate_bias_lap  = args.gate_bias_lap,
        lap_eigvals    = lap_eigvals,
    )







    model = HPEHGTModel(config=cfg, metadata=data.metadata()).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    # Simple step‐LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.1)

    loss_fn = nn.CrossEntropyLoss().to(device)

    Train_Loss, Train_macro_f1, Train_micro_f1 = [], [], []
    Val_Loss, Val_macro_f1, Val_micro_f1 = [], [], []
    Train_time, Val_time = [], []
    sum_train_time = sum_val_time = 0.0

    for i in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_ma, tr_mi = train(model, data, original_A, y, train_mask, deg, loss_fn, optimizer)
        dt_tr = time.time() - t0
        Train_time.append(dt_tr); sum_train_time += dt_tr
        Train_Loss.append(tr_loss); Train_macro_f1.append(tr_ma); Train_micro_f1.append(tr_mi)

        v0 = time.time()
        v_loss, v_ma, v_mi = validate(model, data, original_A, y, val_mask, deg, loss_fn)
        dt_v = time.time() - v0
        Val_time.append(dt_v); sum_val_time += dt_v
        Val_Loss.append(v_loss); Val_macro_f1.append(v_ma); Val_micro_f1.append(v_mi)

        tm, tmi, _ = test(model, data, original_A, y, test_mask, deg)
        if i % 10 == 0:
          print(
            f"Epoch {i:03d} || "
            f"train loss:{float(tr_loss):.3f} time:{dt_tr:.4f}, "
            f"ma_f1:{tr_ma*100:.2f}%, mi_f1:{tr_mi*100:.2f}%"
            f" || val loss:{float(v_loss):.3f} time:{dt_v:.4f}, "
            f"ma_f1:{v_ma*100:.2f}%, mi_f1:{v_mi*100:.2f}%"
            f" || test ma:{tm*100:.2f}%, mi:{tmi*100:.2f}%"
        )
        lr_scheduler.step()

    # final plots + embedding
    draw_loss(Train_Loss, len(Train_Loss), args.dataset, 'Train')
    draw_acc(Train_macro_f1, len(Train_macro_f1), args.dataset, 'Train_macro_f1')
    draw_acc(Train_micro_f1, len(Train_micro_f1), args.dataset, 'Train_micro_f1')
    draw_loss(Val_Loss, len(Val_Loss), args.dataset, 'Val')
    draw_acc(Val_macro_f1, len(Val_macro_f1), args.dataset, 'Val_macro_f1')
    draw_acc(Val_micro_f1, len(Val_micro_f1), args.dataset, 'Val_micro_f1')
    draw_time_per_epoch(Train_time, Val_time, args.dataset)
    draw_cumulative_time(Train_time, Val_time, args.dataset)
    # make sure the time plots are finished before drawing the embedding
    import matplotlib.pyplot as plt
    plt.close('all')      # or plt.figure()
    model.eval()
    with torch.no_grad():
        _, _, embs, _, _ = model(data, original_A, deg)
    visualize_embedding(embs[test_mask].cpu(), y[test_mask].cpu(), name=args.dataset + '_emb')

    print(f"Avg train time/epoch: {sum_train_time/args.epochs:.4f}s")
    print(f"Avg val time/epoch:   {sum_val_time/args.epochs:.4f}s")
        # final test metrics
    macro, micro, _ = test(model, data, original_A, y, test_mask, deg)
    print(f"test_macro_f1:{macro*100:.2f}")
    print(f"test_micro_f1:{micro*100:.2f}")

if __name__ == "__main__":
    main()
