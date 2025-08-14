import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
import math


class Single_Attention_layer(nn.Module):
    def __init__(self, config):
        super().__init__()

        
        self.num_metapaths = config.num_metapaths
        self.svd_dim = config.svd_dim
        self.hid_dim = config.hidden_dim

        
        
        #************ added this for variant B gated normalization *****
        self.lap_dim = config.lap_dim
        self.meta_dim      = self.num_metapaths * self.svd_dim

        # ──******* Eigenvalue‐Informed Gating: eigenvalues for LapPE dims ──
        self.lap_eigvals = config.lap_eigvals


        #********************************** end Eigenvalue‐Informed Gating *****
        total_pe           = self.meta_dim + self.lap_dim
        self.qk_input_dim  = self.hid_dim + total_pe

        #*******************************************



        
        
        self.heads = config.num_heads
        self.dropout = config.dropout
        self.bias = config.bias

        # include both SVD‐PE and (new) Laplacian PE dims
        total_pe = self.num_metapaths * self.svd_dim + config.lap_dim
        self.Q_lin = nn.Linear(in_features=total_pe + self.hid_dim, out_features=self.hid_dim // self.heads, bias=self.bias)
        self.K_lin = nn.Linear(in_features=total_pe + self.hid_dim, out_features=self.hid_dim // self.heads, bias=self.bias)


        self.V_lin = nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim // self.heads,
                               bias=self.bias)
        # one small gate‐MLP for the meta‐path part, one for the Laplacian part, for Q and for K:
        self.gate_meta_Q = nn.Linear(self.meta_dim,  self.meta_dim,  bias=True)


        #****── Eigenvalue‐Informed: LapPE gate now accepts [LapPE; eigvals] → lap_dim
        self.gate_lap_Q  = nn.Linear(self.lap_dim * 2,   self.lap_dim,   bias=True)
        #*************************************

        self.gate_meta_K = nn.Linear(self.meta_dim,  self.meta_dim,  bias=True)


        # ***── Eigenvalue‐Informed: LapPE gate now accepts [LapPE; eigvals] → lap_dim
        self.gate_lap_K  = nn.Linear(self.lap_dim * 2,   self.lap_dim,   bias=True)
        #***********************************


        # initialize biases so σ(bias)=desired starting weight
        nn.init.constant_(self.gate_meta_Q.bias, config.gate_bias_meta)
        nn.init.constant_(self.gate_meta_K.bias,  config.gate_bias_meta)
        nn.init.constant_(self.gate_lap_Q.bias,  config.gate_bias_lap)
        nn.init.constant_(self.gate_lap_K.bias,  config.gate_bias_lap)




    def forward(self, input_x, pe_Q, pe_K):

        #******** option 3: Separate gating for meta‐path vs LapPE ********************#
        # 1) split out sub‐vectors
        Q_meta, Q_lap = pe_Q[:, :self.meta_dim], pe_Q[:, self.meta_dim:]
        K_meta, K_lap = pe_K[:, :self.meta_dim], pe_K[:, self.meta_dim:]

        # 2) optional pre‐gate normalization for stability
        Q_meta_n = F.layer_norm(Q_meta, Q_meta.shape[1:])
        Q_lap_n  = F.layer_norm(Q_lap,  Q_lap.shape[1:])
        K_meta_n = F.layer_norm(K_meta, K_meta.shape[1:])
        K_lap_n  = F.layer_norm(K_lap,  K_lap.shape[1:])

        # 3) per‐block gates
        g_meta_Q = torch.sigmoid(self.gate_meta_Q(Q_meta_n))  # (N, meta_dim)
        
        g_meta_K = torch.sigmoid(self.gate_meta_K(K_meta_n))
        eig_expand = self.lap_eigvals.unsqueeze(0).expand(input_x.size(0), -1)  # (N, lap_dim)
        lap_in_Q   = torch.cat([Q_lap_n, eig_expand], dim=1)                    # (N, 2*lap_dim)
        lap_in_K   = torch.cat([K_lap_n, eig_expand], dim=1)

        g_lap_Q    = torch.sigmoid(self.gate_lap_Q(lap_in_Q))
        g_lap_K    = torch.sigmoid(self.gate_lap_K(lap_in_K))
        #**************************************************#



        # 4) debug log once
        if not hasattr(self, "_gate_logged"):
            print(f"[SepGate] meta_Q:μ={g_meta_Q.mean():.3f},σ={g_meta_Q.std():.3f} | "f"lap_Q:μ={g_lap_Q.mean():.3f},σ={g_lap_Q.std():.3f}")
            self._gate_logged = True

        # 5) apply gates
        Q_meta, Q_lap = Q_meta * g_meta_Q, Q_lap * g_lap_Q
        K_meta, K_lap = K_meta * g_meta_K, K_lap * g_lap_K

        # 6) reassemble
        pe_Q = torch.cat([Q_meta, Q_lap], dim=1)
        pe_K = torch.cat([K_meta, K_lap], dim=1)








        #********************** end option 3 ***************







        # ── Post-gate normalization for stability
        pe_Q = F.layer_norm(pe_Q, pe_Q.shape[1:])
        pe_K = F.layer_norm(pe_K, pe_K.shape[1:])






        # now concatenate
        x_Q = torch.cat((input_x, pe_Q), dim=-1)
        x_K = torch.cat((input_x, pe_K), dim=-1)
#*******************************************************************
        Q = self.Q_lin(x_Q)  # (N,dh)
        K = self.K_lin(x_K)  # (N,dh)
        V = self.V_lin(input_x)  # (N,dh)
        QKT = Q @ K.t()
        # apply per-head scaling before dropout & softmax

        QKT = F.dropout(QKT, p=self.dropout, training=self.training)

        attn = F.softmax(QKT / math.sqrt(self.hid_dim // self.heads), dim=-1)
        out = attn @ V
        return out


class GraphTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hid_dim = config.hidden_dim
        self.heads = config.num_heads
        self.dropout = config.dropout

        self.attn_layers_list = nn.Sequential(*[Single_Attention_layer(config) for _ in range(self.heads)])

        self.cat_heads_out = nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim)


        self.norm1 = nn.LayerNorm(self.hid_dim)
        self.fnn_1 = nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim)

        self.norm2 = nn.LayerNorm(self.hid_dim)
        self.fnn_2 = nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim)

    def forward(self, x, pe_Q, pe_K, deg):
        head_outs_list = []
        for i, blocks in enumerate(self.attn_layers_list):
            head_outs_list.append(blocks(x, pe_Q, pe_K))
        head_outs_list = torch.stack(head_outs_list, dim=0)
        head_outs = head_outs_list.transpose(0, 1).reshape(-1, self.hid_dim).contiguous()
        attn_x = self.cat_heads_out(head_outs)
        attn_x = F.relu(attn_x)

        
    
        deg_sqrt = torch.sqrt(deg + 1e-6).unsqueeze(1)    # (N,1)
        x_1 = x + attn_x / deg_sqrt


        x_1 = self.norm1(x_1)
        # FNN
        x_2 = self.fnn_1(x_1)
        x_2 = self.fnn_2(x_2)


        x_2 = x_1 + x_2
        out = self.norm2(x_2)

        return out
