"""Gradient- and attention-based explanations (CAM family) for fragment and atom graphs,
plus RDKit drawings.

Three per-fragment score functions, all with signature fn(spec, data, device) -> tensor[n]:
  frag_gradcam   Grad-CAM (Selvaraju et al. 2017; for graphs, Pope et al. 2019) on the
                 fragment GIN's last-layer fragment embeddings.
  attention      the fragment GIN's attention-readout weights (where the model pools from).
  atom_gradcam   Grad-CAM on an atom-level GIN trained on the same split, summed over each
                 fragment's atoms. Lets atom- and fragment-level models be compared on the
                 same units.

Scores are signed toward the PREDICTED class: for a negative prediction (or a regression
value below the threshold) the gradient of the negated output is used, so a high score
always means "supports this prediction".

The classic CAM of Zhou et al. (2016) needs global average pooling followed by a linear
layer; the fragment GIN uses attention pooling and an MLP head, so Grad-CAM is used instead.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import softmax as segment_softmax


# ----------------------------------------------------------------------------- fragment GIN

def _one_graph(data, device):
    d = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr).to(device)
    d.batch = torch.zeros(d.x.size(0), dtype=torch.long, device=device)
    return d


def _signed(spec, raw_out):
    """Output oriented toward the predicted class."""
    pred_pos = spec.score(raw_out.detach()).item() >= spec.threshold
    return raw_out.sum() if pred_pos else -raw_out.sum()


def gradcam(h, target):
    """Grad-CAM over nodes: channel weights = mean gradient over nodes; score = ReLU(sum_c a_c h_vc).
    Returns the pre-ReLU map (used for ranking, avoids ties at zero)."""
    grads = torch.autograd.grad(target, h)[0]
    alpha = grads.mean(dim=0)
    return (h * alpha).sum(dim=1).detach().cpu()


def frag_gradcam(spec, data, device):
    gin = spec.gin.eval()
    with torch.enable_grad():
        h, g = gin.embed(_one_graph(data, device))
        return gradcam(h, _signed(spec, gin.head(g)))


@torch.no_grad()
def attention(spec, data, device):
    gin = spec.gin.eval()
    d = _one_graph(data, device)
    h, _ = gin.embed(d)
    gate = gin.readout.gate_nn(h).view(-1)
    return segment_softmax(gate, d.batch).cpu()


# ----------------------------------------------------------------------------- atom GIN

def atom_graph(smiles, y=0.0):
    """PyG's standard RDKit atom graph (9 categorical atom features, 3 bond features).
    Atom order matches Chem.MolFromSmiles(smiles). Returns None if featurization fails."""
    from torch_geometric.utils import from_smiles
    try:
        d = from_smiles(smiles)
    except Exception:
        return None
    if d.x is None or d.x.size(0) == 0:
        return None
    d.y = torch.tensor([float(y)], dtype=torch.float)
    return d


def _vocab_sizes():
    from torch_geometric.utils.smiles import e_map, x_map
    return [len(v) for v in x_map.values()], [len(v) for v in e_map.values()]


class CatEncoder(nn.Module):
    """Sum of one embedding per categorical column (like OGB's AtomEncoder)."""

    def __init__(self, sizes, dim):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(s + 1, dim) for s in sizes)

    def forward(self, x):
        x = x.long()
        return sum(emb(x[:, i].clamp(max=emb.num_embeddings - 1)) for i, emb in enumerate(self.embs))


class AtomGIN(nn.Module):
    def __init__(self, hidden=128, layers=4, dropout=0.2):
        super().__init__()
        xs, es = _vocab_sizes()
        self.atom_enc = CatEncoder(xs, hidden)
        self.convs, self.norms, self.bond_encs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp))
            self.norms.append(nn.BatchNorm1d(hidden))
            self.bond_encs.append(CatEncoder(es, hidden))
        self.readout = AttentionalAggregation(gate_nn=nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.dropout = dropout

    def embed(self, data):
        h = self.atom_enc(data.x)
        for conv, norm, be in zip(self.convs, self.norms, self.bond_encs):
            e = be(data.edge_attr) if data.edge_attr is not None and data.edge_attr.numel() else None
            if e is None:
                e = torch.zeros(data.edge_index.size(1), h.size(1), device=h.device)
            h = h + F.dropout(F.relu(norm(conv(h, data.edge_index, e))), self.dropout, self.training)
        return h, self.readout(h, data.batch)

    def forward(self, data):
        return self.head(self.embed(data)[1]).squeeze(-1)


def build_atom_splits(frag_splits):
    """Atom graphs for the same molecules and the same scaffold split as the fragment graphs."""
    out = []
    for split in frag_splits:
        gs = [atom_graph(d.smiles, float(d.y)) for d in split]
        out.append([g for g in gs if g is not None])
    return out


def train_atom_gin(name, task, splits, seed, device, ckpt_dir="checkpoints", epochs=150, patience=25, lr=1e-3):
    """Same protocol as frag_gin.train_one; returns (valid metric, test metric, model, mu, sd)."""
    from frag_gin import evaluate
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    train, valid, test = splits
    ys = torch.cat([d.y for d in train])
    mu, sd = (ys.mean().item(), ys.std().item()) if task == "regression" else (0.0, 1.0)
    tl = DataLoader(train, batch_size=64, shuffle=True, drop_last=True)
    vl, te = DataLoader(valid, batch_size=256), DataLoader(test, batch_size=256)
    model = AtomGIN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    better = (lambda a, b: a > b) if task == "classification" else (lambda a, b: a < b)
    best, best_state, wait = None, None, 0
    for _ in range(epochs):
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
    model.eval()
    test_metric = evaluate(model, te, task, mu, sd, device)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"state_dict": best_state, "mu": mu, "sd": sd, "valid": best, "test": test_metric},
               f"{ckpt_dir}/{name}_atomgin_seed{seed}.pt")
    return best, test_metric, model, mu, sd


def load_atom_gin(path, device):
    """Reload a trained atom GIN. Returns (model, mu, sd, valid metric, test metric);
    the metrics are None for checkpoints saved before they were recorded."""
    ckpt = torch.load(path, weights_only=False, map_location=device)
    model = AtomGIN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mu"], ckpt["sd"], ckpt.get("valid"), ckpt.get("test")


def atom_gradcam_scores(atom_spec, smiles, device):
    """Per-atom Grad-CAM for the atom GIN (pre-ReLU), oriented toward its own predicted class."""
    g = atom_graph(smiles)
    d = g.to(device)
    d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=device)
    model = atom_spec.gin.eval()
    with torch.enable_grad():
        h, pooled = model.embed(d)
        return gradcam(h, _signed(atom_spec, model.head(pooled)))


def make_atom_gradcam(atom_spec):
    """Returns fn(spec, data, device) -> per-fragment scores: atom Grad-CAM summed over each
    fragment's atoms. `data` must come from canonical_graph() so atom indices line up."""
    def fn(spec, data, device):
        a = atom_gradcam_scores(atom_spec, data.smiles, device)
        return torch.stack([a[list(group)].sum() for group in data.atom_groups])
    return fn


# ----------------------------------------------------------------------------- drawing
#
# Two figure types:
#   explanation_grid   one row per molecule, one panel per method. Fragments are shown as
#                      whole units: atoms AND the bonds inside each fragment are colored,
#                      the bonds BRICS cut are left gray, and each fragment is numbered F1..Fn.
#   fragment_panels    one row per molecule: the molecule (fragments numbered and outlined by
#                      color) followed by each fragment drawn on its own, with its attachment
#                      points, colored by importance and captioned with every method's score.
#
# Colors: red supports the prediction, blue opposes it, scaled per molecule.

CUT_BOND_COLOR = (0.75, 0.75, 0.75)


def _colors(values, cmap="RdBu_r"):
    import matplotlib
    v = np.asarray(values, dtype=float)
    lim = max(np.abs(v).max(), 1e-8)
    cm = matplotlib.colormaps[cmap]
    return [tuple(cm(0.5 + 0.45 * x / lim)[:3]) for x in v]


def _frag_index(atom_groups):
    return {a: f for f, g in enumerate(atom_groups) for a in g}


def _cut_bonds(mol, atom_groups):
    """Bonds whose two atoms sit in different fragments (the BRICS cuts)."""
    fi = _frag_index(atom_groups)
    return [b.GetIdx() for b in mol.GetBonds() if fi[b.GetBeginAtomIdx()] != fi[b.GetEndAtomIdx()]]


def _png(drawer):
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _drawer(size, legend_size=15):
    from rdkit.Chem.Draw import rdMolDraw2D
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions()
    o.legendFontSize = legend_size
    o.annotationFontScale = 0.8
    o.fillHighlights = True
    o.continuousHighlight = True
    o.highlightBondWidthMultiplier = 14
    return d


def draw_fragment_map(smiles, atom_groups, frag_scores=None, atom_scores=None, keep=None,
                      size=(340, 280), legend="", number=True):
    """Draw a molecule with fragment structure visible.

    frag_scores: one value per fragment -> the fragment's atoms and internal bonds share its color.
    atom_scores: one value per atom (atom-level Grad-CAM) -> atoms colored individually; each
                 internal bond gets the mean of its two atoms.
    keep:        list of fragment indices -> those fragments in red, others uncolored (GSPELL).
    Cut bonds stay gray in every case, so fragment boundaries are visible."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = Chem.Mol(Chem.MolFromSmiles(smiles))
    fi = _frag_index(atom_groups)
    if keep is not None:
        frag_scores = [1.0 if f in set(keep) else 0.0 for f in range(len(atom_groups))]
    if frag_scores is not None:
        fc = _colors(frag_scores)
        atom_col = {a: fc[fi[a]] for a in range(mol.GetNumAtoms())}
        if keep is not None:   # unselected fragments: no color at all
            atom_col = {a: c for a, c in atom_col.items() if fi[a] in set(keep)}
    else:
        ac = _colors(atom_scores)
        atom_col = {a: ac[a] for a in range(mol.GetNumAtoms())}

    bond_col = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if fi[i] != fi[j]:
            bond_col[b.GetIdx()] = CUT_BOND_COLOR
        elif i in atom_col and j in atom_col:
            if atom_scores is not None and frag_scores is None:
                bond_col[b.GetIdx()] = tuple((np.array(atom_col[i]) + np.array(atom_col[j])) / 2)
            else:
                bond_col[b.GetIdx()] = atom_col[i]

    if number:   # label each fragment once, on its first atom
        for f, g in enumerate(atom_groups):
            mol.GetAtomWithIdx(int(g[0])).SetProp("atomNote", f"F{f + 1}")

    d = _drawer(size)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, legend=legend,
        highlightAtoms=list(atom_col), highlightAtomColors=atom_col,
        highlightBonds=list(bond_col), highlightBondColors=bond_col,
        highlightAtomRadii={a: 0.4 for a in atom_col})
    return _png(d)


def fragment_mols(smiles, atom_groups):
    """Each fragment as its own molecule with dummy atoms (*) where it was attached."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    cuts = _cut_bonds(mol, atom_groups)
    pieces = Chem.FragmentOnBonds(mol, cuts, addDummies=True, dummyLabels=[(0, 0)] * len(cuts)) if cuts else mol
    piece_atoms = Chem.GetMolFrags(pieces, asMols=False, sanitizeFrags=False)
    owner = {a: p for p, atoms in enumerate(piece_atoms) for a in atoms}
    out = []
    for g in atom_groups:
        atoms = piece_atoms[owner[int(g[0])]]
        smi = Chem.MolFragmentToSmiles(pieces, atomsToUse=list(atoms), canonical=False, kekuleSmiles=False)
        m = Chem.MolFromSmiles(smi, sanitize=False)
        m.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                         ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY, catchErrors=True)
        out.append(m)
    return out


def draw_fragment_tile(frag_mol, color, size=(220, 200), legend=""):
    """One fragment at a fixed bond length (so a single atom isn't blown up to fill the tile)."""
    from rdkit.Chem.Draw import rdMolDraw2D
    atoms = [a.GetIdx() for a in frag_mol.GetAtoms() if a.GetAtomicNum() != 0]
    bonds = [b.GetIdx() for b in frag_mol.GetBonds()
             if b.GetBeginAtom().GetAtomicNum() != 0 and b.GetEndAtom().GetAtomicNum() != 0]
    d = _drawer(size, legend_size=13)
    d.drawOptions().padding = 0.15
    d.drawOptions().fixedBondLength = 28
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, frag_mol, legend=legend, kekulize=False,
        highlightAtoms=atoms, highlightAtomColors={a: color for a in atoms},
        highlightBonds=bonds, highlightBondColors={b: color for b in bonds},
        highlightAtomRadii={a: 0.4 for a in atoms})
    return _png(d)


def _images(pngs):
    import io
    from PIL import Image
    return [Image.open(io.BytesIO(p)).convert("RGB") for p in pngs]


def _font(size):
    from PIL import ImageFont
    for path in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _caption(img, lines, size=14, bold_first=True, pad=6):
    """Image with text lines underneath (readable size, unlike RDKit legends that shrink)."""
    from PIL import Image, ImageDraw
    f, fb = _font(size), _font(size + 1)
    line_h = size + 5
    out = Image.new("RGB", (img.width, img.height + pad + line_h * len(lines)), "white")
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        font = fb if (i == 0 and bold_first) else f
        w = draw.textlength(line, font=font)
        draw.text(((img.width - w) / 2, img.height + pad + i * line_h), line, fill="black", font=font)
    return out


def _stack_rows(rows, pad=10, bg="white"):
    from PIL import Image
    width = max(sum(im.width for im in r) + pad * (len(r) - 1) for r in rows)
    height = sum(max(im.height for im in r) for r in rows) + pad * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), bg)
    y = 0
    for r in rows:
        x = 0
        for im in r:
            canvas.paste(im, (x, y))
            x += im.width + pad
        y += max(im.height for im in r) + pad
    return canvas


def explanation_grid(records, atom_spec=None, device=None, n=4, out_path="figures/cam_examples.png"):
    """Rows: molecules. Columns: atom Grad-CAM, fragment Grad-CAM, GSPELL YES fragments.
    Atom Grad-CAM comes from the stored atom_gradcam per-atom scores if atom_spec is None."""
    rows = []
    for _, r in records.head(n).iterrows():
        smi, groups = r["smiles"], r["atom_groups"]
        panels = []
        if atom_spec is not None:
            a = atom_gradcam_scores(atom_spec, smi, device).numpy()
            panels.append(draw_fragment_map(smi, groups, atom_scores=a, legend="Grad-CAM, atom GIN"))
        panels.append(draw_fragment_map(smi, groups, frag_scores=r["frag_gradcam_scores"], legend="Grad-CAM, fragment GIN"))
        panels.append(draw_fragment_map(smi, groups, keep=r["gspell_keep"], legend="GSPELL: YES fragments"))
        rows.append(_images(panels))
    grid = _stack_rows(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    grid.save(out_path)
    return grid


SCORE_LABELS = {"shapley": "Shapley", "frag_gradcam": "fCAM", "atom_gradcam": "aCAM", "attention": "attn", "gnnexp": "GNNExp"}


def fragment_panels(records, n=4, color_by="frag_gradcam", out_path="figures/fragment_panels.png",
                    max_per_row=6):
    """Per molecule: the molecule with numbered fragments, then each fragment on its own,
    colored by `color_by` scores and captioned with GSPELL's verdict and every method's score."""
    import textwrap
    from rdkit import Chem
    rows = []
    for _, r in records.head(n).iterrows():
        smi, groups = r["smiles"], r["atom_groups"]
        key = color_by if isinstance(r.get(f"{color_by}_scores"), (list, tuple)) else "frag_gradcam"
        scores = np.asarray(r[f"{key}_scores"], dtype=float)
        mol_png = draw_fragment_map(smi, groups, frag_scores=scores, size=(380, 300),
                                    legend=f"GIN score {r['gin_score']:.2f} | colored by {SCORE_LABELS.get(key, key)}")
        tiles = []
        cols = _colors(scores)
        keep = set(r["gspell_keep"])
        for f, (fm, c) in enumerate(zip(fragment_mols(smi, groups), cols)):
            vals = [f"{SCORE_LABELS[k]} {r[f'{k}_scores'][f]:+.2f}" for k in SCORE_LABELS
                    if isinstance(r.get(f"{k}_scores"), (list, tuple))]
            label = r["fragments"][f].split(" (", 1)[-1].rstrip(")") if " (" in r["fragments"][f] else ""
            ci = "counterion" in r and r["counterion"] and r["counterion"][f]
            flip = isinstance(r.get("removal_flips"), (list, tuple)) and r["removal_flips"][f]
            head = f"F{f + 1}{' [ion]' if ci else ''}  GSPELL {'YES' if f in keep else 'no'}{'  removal flips' if flip else ''}"
            lines = [head, textwrap.shorten(label, 36, placeholder="...")] + ["   ".join(vals[i:i + 2]) for i in range(0, len(vals), 2)]
            tiles.append(_caption(_images([draw_fragment_tile(fm, c, size=(230, 190))])[0], lines))
        imgs = _images([mol_png]) + tiles
        # wrap long fragment lists onto continuation rows
        rows.append(imgs[:1 + max_per_row])
        for k in range(1 + max_per_row, len(imgs), max_per_row):
            from PIL import Image
            rows.append([Image.new("RGB", imgs[0].size, "white")] + imgs[k:k + max_per_row])
    grid = _stack_rows(rows, pad=12)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    grid.save(out_path)
    return grid


# ----------------------------------------------------------------------------- agreement

def agreement_table(records, methods=("gnnexp", "frag_gradcam", "attention", "atom_gradcam")):
    """Per method: mean Jaccard of its size-matched top-k with GSPELL's YES set, and mean
    Spearman correlation of its fragment scores with fragment Grad-CAM."""
    import pandas as pd
    from scipy.stats import spearmanr
    rows = []
    for m in methods:
        if f"{m}_scores" not in records:
            continue
        jac, rho = [], []
        for _, r in records.iterrows():
            a = set(r["gspell_keep"])
            k = max(1, len(a))
            b = set(np.argsort(-np.asarray(r[f"{m}_scores"]))[:k].tolist())
            if a:
                jac.append(len(a & b) / len(a | b))
            if m != "frag_gradcam" and r["n_fragments"] >= 3:
                rho.append(spearmanr(r[f"{m}_scores"], r["frag_gradcam_scores"]).correlation)
        rows.append({"method": m, "jaccard_with_GSPELL": np.nanmean(jac) if jac else np.nan,
                     "spearman_with_frag_gradcam": np.nanmean(rho) if rho else np.nan})
    return pd.DataFrame(rows)


def counterion_usage(records, methods=("gspell", "llm", "gnnexp", "gnnexp_k", "frag_gradcam", "attention", "atom_gradcam")):
    """Ionic mode only. Over molecules that have counter-ion nodes: how often each method's
    explanation includes a counter-ion, and the share of counter-ion nodes it selects.
    Compare with the base rate (share of all fragments that are counter-ions)."""
    import pandas as pd
    salts = records[records["counterion"].apply(any)]
    if len(salts) == 0:
        return pd.DataFrame()
    base = np.mean([sum(c) / len(c) for c in salts["counterion"]])
    rows = []
    for m in methods:
        if f"{m}_keep" not in salts:
            continue
        incl, share = [], []
        for _, r in salts.iterrows():
            ci = {i for i, c in enumerate(r["counterion"]) if c}
            keep = set(r[f"{m}_keep"])
            incl.append(bool(keep & ci))
            share.append(len(keep & ci) / len(ci))
        rows.append({"method": m, "molecules_with_counterions": len(salts),
                     "pct_explanations_including_counterion": 100 * np.mean(incl),
                     "pct_counterion_nodes_selected": 100 * np.mean(share),
                     "base_rate_counterion_fraction_%": 100 * base})
    return pd.DataFrame(rows)
