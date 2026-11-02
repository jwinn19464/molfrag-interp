"""Fragment-graph GIN for molecular property prediction.

Fragment features -> MLP compression -> GINE layers (edge features = broken-bond type)
-> attention readout -> prediction head. `embed()` returns the fragment (node) and
molecule (graph) embeddings that the GSPELL projector consumes.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv
from torch_geometric.nn.aggr import AttentionalAggregation
from sklearn.metrics import roc_auc_score


class FragGIN(nn.Module):
    def __init__(self, in_dim, edge_dim=5, hidden=128, layers=3, dropout=0.2):
        super().__init__()
        self.compress = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden))
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.readout = AttentionalAggregation(gate_nn=nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.dropout = dropout

    def embed(self, data):
        """Fragment (node) embeddings and molecule (graph) embedding."""
        h = self.compress(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.relu(norm(conv(h, data.edge_index, data.edge_attr))), self.dropout, self.training)
        return h, self.readout(h, data.batch)

    def forward(self, data):
        return self.head(self.embed(data)[1]).squeeze(-1)


def load_graphs(name, graph_dir="frag_graphs"):
    return [torch.load(os.path.join(graph_dir, f"{name}_{s}.pt"), weights_only=False)
            for s in ("train", "valid", "test")]


def load_checkpoint(path, device):
    ckpt = torch.load(path, weights_only=False, map_location=device)
    model = FragGIN(in_dim=ckpt["in_dim"], edge_dim=ckpt.get("edge_dim", 5)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate(model, loader, task, mu, sd, device):
    model.eval()
    preds, ys = [], []
    for b in loader:
        b = b.to(device)
        out = model(b)
        preds.append(torch.sigmoid(out) if task == "classification" else out * sd + mu)
        ys.append(b.y)
    p, y = torch.cat(preds).cpu().numpy(), torch.cat(ys).cpu().numpy()
    return roc_auc_score(y, p) if task == "classification" else float(np.sqrt(((p - y) ** 2).mean()))


def train_one(name, task, seed, device, graph_dir="frag_graphs", ckpt_dir="checkpoints",
              epochs=150, patience=25, lr=1e-3):
    """Train with early stopping on the validation metric; returns (best valid, test, epochs run)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    train, valid, test = load_graphs(name, graph_dir)
    ys = torch.cat([d.y for d in train])
    mu, sd = (ys.mean().item(), ys.std().item()) if task == "regression" else (0.0, 1.0)
    tl = DataLoader(train, batch_size=64, shuffle=True, drop_last=True)
    vl, te = DataLoader(valid, batch_size=256), DataLoader(test, batch_size=256)

    in_dim, edge_dim = train[0].x.size(1), train[0].edge_attr.size(1)   # 523/5 default, 525/6 ionic mode
    model = FragGIN(in_dim=in_dim, edge_dim=edge_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    better = (lambda a, b: a > b) if task == "classification" else (lambda a, b: a < b)
    best, best_state, wait, epoch = None, None, 0, 0
    for epoch in range(epochs):
        model.train()
        for b in tl:
            b = b.to(device)
            opt.zero_grad()
            out = model(b)
            loss = (F.binary_cross_entropy_with_logits(out, b.y) if task == "classification"
                    else F.mse_loss(out, (b.y - mu) / sd))
            loss.backward()
            opt.step()
        v = evaluate(model, vl, task, mu, sd, device)
        if best is None or better(v, best):
            best, wait = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    test_metric = evaluate(model, te, task, mu, sd, device)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"state_dict": best_state, "mu": mu, "sd": sd, "in_dim": in_dim, "edge_dim": edge_dim},
               os.path.join(ckpt_dir, f"{name}_gin_seed{seed}.pt"))
    return best, test_metric, epoch + 1
