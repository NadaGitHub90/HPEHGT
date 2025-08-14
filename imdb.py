import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

from HPEHGT.hpeutils.data import get_dataset, train_val_test_split
import torch.nn as nn
from torch_geometric.utils import degree
from HPEHGT.hpeutils.utils import set_random_seed
from HPEHGT.hpeutils.draw import draw_loss, draw_acc, visualize_embedding
from HPEHGT.hpeutils.draw import draw_time_per_epoch, draw_cumulative_time
from HPEHGT.model.IMDB.hpehgt import HPEHGTModel
from sklearn.metrics import f1_score
import time
from dataclasses import dataclass
import math

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def load_args():
    parser = argparse.ArgumentParser(description='HPEHGT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=52, help='random seed')
    parser.add_argument('--dataset', type=str, default='IMDB', help='name of dataset')
    parser.add_argument('--hidden-dim', type=int, default=64, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout-rate")
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
    parser.add_argument('--subset_size', type=int, default=None,
                        help='If set, randomly sample this many MOVIE nodes (and incident edges)')
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
        torch.save(pe_Q, 'IMDB_Q.pth')
        torch.save(pe_K, 'IMDB_K.pth')
        marco_f1 = f1_score(y_true=y[test_mask], y_pred=pred[test_mask], average='macro')
        mirco_f1 = f1_score(y_true=y[test_mask], y_pred=pred[test_mask], average='micro')
    return marco_f1, mirco_f1, embs


def main():
    global args
    args = load_args()
    # ── If the user asked for ACM, hand off to the specialized script ──
    if args.dataset.upper() == 'ACM':
        from HPEHGT.acm import main as _acm_main
        return _acm_main()
    print(args)

    set_random_seed(args.seed)
    dataset = get_dataset(args.dataset, True)
    data = dataset[0]
    print(f"→ Loaded full dataset: {data}")

    # ── Optional subgraph sampling on MOVIE nodes ──
    if args.subset_size is not None:
        total_movies = data['movie'].num_nodes
        N = args.subset_size
        assert N <= total_movies, \
            f"subset_size ({N}) cannot exceed total movie nodes ({total_movies})"
        print(f"→ [DEBUG] Sampling subset_size={N} movie nodes out of {total_movies}")

        perm = torch.randperm(total_movies, device='cpu')
        keep = perm[:N]
        movie_mask = torch.zeros(total_movies, dtype=torch.bool, device='cpu')
        movie_mask[keep] = True

        old2new = torch.full((total_movies,), -1, dtype=torch.long, device='cpu')
        old2new[keep] = torch.arange(N, dtype=torch.long, device='cpu')

        for key in ['x', 'y', 'train_mask', 'val_mask', 'test_mask']:
            data['movie'][key] = data['movie'][key][keep]

        for et in data.edge_types:
            src, dst = data[et].edge_index
            keep_edge = torch.ones(src.size(0), dtype=torch.bool, device=src.device)
            if et[0] == 'movie':
                km = movie_mask[src.cpu()].to(src.device)
                src = old2new[src.cpu()].to(src.device)
                keep_edge &= km
            if et[2] == 'movie':
                km = movie_mask[dst.cpu()].to(dst.device)
                dst = old2new[dst.cpu()].to(dst.device)
                keep_edge &= km
            data[et].edge_index = torch.stack(
                [src[keep_edge], dst[keep_edge]], dim=0
            )
            print(f"   [DEBUG] Edge type {et}: kept {data[et].edge_index.size(1)} edges")

        print(f"→ [DEBUG] After subsampling & reindexing: {data}")

    # ────────────────────────────────────────────────

    # build metapath adjacencies
    node_types, edge_types = data.metadata()
    num_node = data['movie'].x.shape[0]
    num_classes = len(data['movie'].y.unique())
    n_director = data['director'].x.shape[0]
    n_actor = data['actor'].x.shape[0]

    original_A = []
    e_m_d = data[('movie','to','director')].edge_index
    adj_m_d = torch.sparse_coo_tensor(e_m_d, torch.ones(e_m_d.shape[1]), (num_node, n_director))
    e_m_a = data[('movie','to','actor')].edge_index
    adj_m_a = torch.sparse_coo_tensor(e_m_a, torch.ones(e_m_a.shape[1]), (num_node, n_actor))

    adj_mam = torch.sparse.mm(adj_m_a, adj_m_a.t()).to_dense()
    adj_mdm = torch.sparse.mm(adj_m_d, adj_m_d.t()).to_dense()

    original_A.append(adj_mam)
    original_A.append(adj_mdm)
    original_A = torch.stack(original_A, dim=0)
    torch.save(original_A, 'IMDB_origin_A.pth')

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

    #***** changes here for eigenvectors changes
    lap_t = torch.from_numpy(lap_evecs).float().to(device)  # (num_nodes, lap_dim)
    # ── Eigenvalue-Informed Gating prep: grab the k nontrivial eigenvalues ──
    lap_eigvals_np = vals[1:k+1]
    lap_eigvals = torch.from_numpy(lap_eigvals_np).float().to(device)  # (lap_dim,)
    #*******************





    print(f"[DEBUG] Lap PE shape: {lap_t.shape}, mean ℓ₂ norm = "f"{lap_t.norm(p=2,dim=1).mean():.3f}", flush=True)
    # overwrite the .npy with the normalized vectors
    np.save('movie_lap_evecs.npy', lap_evecs)
    print(f"[LapPE] saved movie_lap_evecs.npy shape={lap_evecs.shape}", flush=True)
    # ── also attach Lap‐PE to data for the model ──
    lap_tensor = torch.from_numpy(lap_evecs).to(device)  # (N, lap_dim)
    data['movie'].lap = lap_tensor
    # ──************************ end LapPE block ──********************



    # degree + splits
    all_A = original_A.sum(dim=0)
    deg = degree(all_A.nonzero().t()[0], num_nodes=num_node).to(device)
    train_mask, val_mask, test_mask = train_val_test_split(
        num_nodes=num_node,
        y=data['movie'].y,
        train_p=0.5,
        val_p=0.25
    )

    data = data.to(device)
    original_A = original_A.to(device)
    y = data['movie'].y.to(device)

    @dataclass
    class IMDBconfig:
        num_nodes: int = num_node
        x_input_dim: int = data['movie'].x.shape[1]
        hidden_dim: int = args.hidden_dim
        svd_dim: int = 16
        classes: int = num_classes
        num_heads: int = args.heads
        dropout: float = args.dropout
        bias: bool = True
        num_blocks: int = args.num_layers
        num_metapaths: int = 2
        num_gnns: int = 2
        lap_dim: int   = args.lap_dim      # ← new: how many Lap-PE dims to expect
        gate_bias_meta: float  = args.gate_bias_meta
        gate_bias_lap:  float  = args.gate_bias_lap

        # ── Eigenvalue-Informed Gating: attach Laplacian eigenvectors ──
        lap_eigvals: torch.Tensor = None    # will attach below
    cfg   = IMDBconfig()
    cfg.lap_eigvals = lap_eigvals
    model = HPEHGTModel(config=cfg, metadata=data.metadata()).to(device)

#**************************************


    # Use decoupled weight decay optimizer
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

