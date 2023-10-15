import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.sparse


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))    #初始化为可训练的参数
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = torch.sparse.mm(G.to_sparse(), x)
        return x


class compute_G(nn.Module):
    def __init__(self, W):
        super(compute_G, self).__init__()
        self.W = Parameter(W)
    def forward(self, DV2_H, invDE_HT_DV2):
        w = torch.diag(self.W)
        invDE_HT_DV2 = invDE_HT_DV2
        G = torch.mm(w, invDE_HT_DV2)
        #G = DV2_H.matmul(G)
        G = torch.mm(DV2_H, G)
        return G

class ResHGNNConv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.W = nn.Linear(in_ft, out_ft, bias=bias)

    def forward(self, X, A, alpha, beta, X0):
        X = A.mm(X0)
        # X = normalize_l2(X)
        Xi = (1 - alpha) * X + alpha * X0 
        X = (1 - beta) * Xi + beta * self.W(Xi)
        return X

class HGNN_weight(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, W, dropout=0.5, momentum=0.2):
        super(HGNN_weight, self).__init__()
        self.dropout = dropout
        self.computeG = compute_G(W)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_ch, n_hid))
        for _ in range(3):
            self.convs.append(ResHGNNConv(n_hid, n_hid))
        self.convs.append(torch.nn.Linear(n_hid, n_class))

        self.hgc2 = HGNN_conv(n_hid, n_class)
        self.batch_normalzation1 = nn.BatchNorm1d(in_ch, momentum=momentum)
        self.batch_normalzation2 = nn.BatchNorm1d(n_hid, momentum=momentum)


    def forward(self, x, DV2_H, invDE_HT_DV2):
        x = self.batch_normalzation1(x)

        G = self.computeG(DV2_H, invDE_HT_DV2)
        lamda, alpha = 0.5, 0.1
        # lamda, alpha = 0.4, 0.4
        x = self.dropout(x)
        x = F.relu(self.convs[0](x))
        x0 = x 
        for i,con in enumerate(self.convs[1:-1]):
            x = self.batch_normalzation2(x)
            x = self.dropout(x)
            beta = math.log(lamda/(i+1)+1)
            x = F.relu(con(x, G, alpha, beta, x0))
        
        x = self.batch_normalzation2(x)
        x = self.dropout(x)
        x = self.hgc2(x, G)
        return x
