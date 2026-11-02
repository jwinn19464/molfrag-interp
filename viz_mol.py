"""Model diagnostics and visual summaries.

  predict_all            test-set scores, labels, molecule and fragment embeddings
  confusion_figure       classification: confusion matrix (counts and row-normalized) + metrics
  parity_figure          regression: predicted vs. measured, with RMSE / MAE / R^2
                         (the regression counterpart of a confusion matrix)
  embedding_figure       t-SNE and UMAP of molecule embeddings, colored by label and by correctness
  fragment_embedding_figure  t-SNE / UMAP of fragment embeddings, colored by functional group
  attention_by_group     mean readout attention per functional group (bar chart)
  attention_maps         molecules drawn with readout-attention weights per fragment
  explanation_heatmap    fragments x methods heatmap for one molecule (GSPELL, Grad-CAM, attention, ...)
  alignment_figure       UMAP of projected soft prompts next to the LLM text embeddings they align to
"""

import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax as segment_softmax

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "font.size": 9})


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")


# ----------------------------------------------------------------------------- predictions

@torch.no_grad()
def predict_all(spec, graphs, device, batch_size=256):
    """Scores (probability or value), labels, molecule embeddings, fragment embeddings,
    fragment readout attention and fragment texts for a list of graphs."""
    gin = spec.gin.eval()
    scores, ys, G, H, A, texts = [], [], [], [], [], []
    for b in DataLoader(graphs, batch_size=batch_size):
        b = b.to(device)
        h, g = gin.embed(b)
        scores.append(spec.score(gin.head(g).squeeze(-1)).cpu())
        ys.append(b.y.cpu())
        G.append(g.cpu()); H.append(h.cpu())
        A.append(segment_softmax(gin.readout.gate_nn(h).view(-1), b.batch).cpu())
        texts += [t for ts in b.fragment_text for t in ts]
    return {"score": torch.cat(scores).numpy(), "y": torch.cat(ys).numpy(),
            "graph_emb": torch.cat(G).numpy(), "frag_emb": torch.cat(H).numpy(),
            "frag_attention": torch.cat(A).numpy(), "frag_text": texts}


def fragment_group(text):
    """First functional-group label from node text 'SMILES (label, label)'."""
    if " (" not in text:
        return "other"
    inner = text.split(" (", 1)[1].rstrip(")")
    if inner.startswith("counter-ion"):
        return "counter-ion"
    return inner.split(",")[0].strip()


# ----------------------------------------------------------------------------- classification / regression

def confusion_figure(pred, title, path, threshold=0.5, labels=("negative", "positive")):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)
    y, p = pred["y"].astype(int), pred["score"]
    yhat = (p >= threshold).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    metrics = {"ROC-AUC": roc_auc_score(y, p), "accuracy": accuracy_score(y, yhat),
               "balanced acc.": balanced_accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
               "recall": recall_score(y, yhat, zero_division=0), "F1": f1_score(y, yhat, zero_division=0)}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, mat, fmt, sub in [(axes[0], cm, "d", "counts"), (axes[1], cmn, ".2f", "row-normalized (recall)")]:
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if cmn[i, j] > 0.6 else "black", fontsize=11)
        ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(sub, fontsize=9)
    fig.suptitle(title + "\n" + "  ".join(f"{k} {v:.3f}" for k, v in metrics.items()), fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    return fig, cm, metrics


def parity_figure(pred, title, path, threshold=None):
    y, p = pred["y"], pred["score"]
    rmse = float(np.sqrt(((p - y) ** 2).mean())); mae = float(np.abs(p - y).mean())
    r2 = 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    ax.scatter(y, p, s=10, alpha=0.6, edgecolors="none")
    if threshold is not None:   # the cut used to turn ESOL into classes for explanation
        ax.axvline(threshold, color="#c0392b", lw=0.8, alpha=0.6)
        ax.axhline(threshold, color="#c0392b", lw=0.8, alpha=0.6)
    ax.set_xlabel("measured"); ax.set_ylabel("predicted")
    ax.set_title(f"{title}\nRMSE {rmse:.3f}  MAE {mae:.3f}  $R^2$ {r2:.3f}", fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    return fig, {"RMSE": rmse, "MAE": mae, "R2": float(r2)}


# ----------------------------------------------------------------------------- embeddings

def _reduce(X, method, seed=0):
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=min(30, max(5, len(X) // 4)), init="pca",
                    random_state=seed).fit_transform(X)
    import umap   # pip install umap-learn
    return umap.UMAP(n_neighbors=min(15, len(X) - 1), min_dist=0.1, random_state=seed).fit_transform(X)


def available_methods():
    try:
        import umap  # noqa: F401
        return ["tsne", "umap"]
    except ImportError:
        return ["tsne"]


def embedding_figure(pred, task, title, path, threshold=0.5, methods=None):
    """Rows: t-SNE / UMAP. Columns: colored by true label (or measured value), and by
    whether the prediction is correct (classification) or by absolute error (regression)."""
    methods = methods or available_methods()
    fig, axes = plt.subplots(len(methods), 2, figsize=(7.4, 3.3 * len(methods)), squeeze=False)
    y, p = pred["y"], pred["score"]
    for r, m in enumerate(methods):
        Z = _reduce(pred["graph_emb"], m)
        ax = axes[r, 0]
        if task == "classification":
            for c, col, lab in [(0, "#3b75af", "label 0"), (1, "#d1495b", "label 1")]:
                ax.scatter(*Z[y == c].T, s=8, alpha=0.7, color=col, label=lab, edgecolors="none")
            ax.legend(frameon=False, fontsize=7, markerscale=2)
        else:
            sc = ax.scatter(*Z.T, s=8, c=y, cmap="viridis", edgecolors="none")
            fig.colorbar(sc, ax=ax, fraction=0.046, label="measured")
        ax.set_title(f"{m.upper()} — true label", fontsize=9)
        ax = axes[r, 1]
        if task == "classification":
            ok = (p >= threshold) == (y >= 0.5)
            ax.scatter(*Z[ok].T, s=8, alpha=0.5, color="#9aa5b1", label="correct", edgecolors="none")
            ax.scatter(*Z[~ok].T, s=12, alpha=0.9, color="#e07b39", label="wrong", edgecolors="none")
            ax.legend(frameon=False, fontsize=7, markerscale=2)
            ax.set_title(f"{m.upper()} — prediction correct?", fontsize=9)
        else:
            sc = ax.scatter(*Z.T, s=8, c=np.abs(p - y), cmap="magma_r", edgecolors="none")
            fig.colorbar(sc, ax=ax, fraction=0.046, label="|error|")
            ax.set_title(f"{m.upper()} — absolute error", fontsize=9)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, path)
    return fig


def fragment_embedding_figure(pred, title, path, top=8, methods=None, max_points=6000, seed=0):
    """Fragment (node) embeddings colored by their main functional-group label."""
    methods = methods or available_methods()
    groups = np.array([fragment_group(t) for t in pred["frag_text"]])
    X = pred["frag_emb"]
    idx = np.arange(len(X))
    if len(idx) > max_points:
        idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
    X, groups = X[idx], groups[idx]
    common = [g for g, _ in Counter(groups).most_common(top)]
    cmap = matplotlib.colormaps["tab10"]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.4 * len(methods), 3.8), squeeze=False)
    for ax, m in zip(axes[0], methods):
        Z = _reduce(X, m, seed)
        rest = ~np.isin(groups, common)
        ax.scatter(*Z[rest].T, s=4, color="#d0d0d0", edgecolors="none", label="other groups")
        for k, g in enumerate(common):
            sel = groups == g
            ax.scatter(*Z[sel].T, s=5, color=cmap(k), edgecolors="none", label=g)
        ax.set_title(m.upper(), fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    axes[0, -1].legend(frameon=False, fontsize=7, markerscale=3, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, path)
    return fig


# ----------------------------------------------------------------------------- attention

def attention_by_group(pred, title, path, top=12, min_count=10):
    """Mean readout attention per functional group, relative to uniform attention within each
    molecule (1.0 = the fragment gets its fair share; > 1 = attended more than average)."""
    import pandas as pd
    # attention relative to 1/n_fragments of its molecule needs molecule sizes: recover from
    # attention sums (each molecule's weights sum to 1) by walking the flat list.
    rel, i = [], 0
    att = pred["frag_attention"]
    sizes = pred.get("n_frag")
    if sizes is None:
        raise ValueError("pass pred from predict_all_with_sizes")
    for n in sizes:
        rel += list(att[i:i + n] * n)
        i += n
    df = pd.DataFrame({"group": [fragment_group(t) for t in pred["frag_text"]], "rel_attention": rel})
    stats = df.groupby("group")["rel_attention"].agg(["mean", "sem", "count"])
    stats = stats[stats["count"] >= min_count].sort_values("mean", ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(4.8, 0.28 * len(stats) + 1.0))
    ax.barh(stats.index[::-1], stats["mean"][::-1], xerr=stats["sem"][::-1], color="#4c78a8", alpha=0.85)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("readout attention relative to uniform")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    return fig, stats


def predict_all_with_sizes(spec, graphs, device):
    pred = predict_all(spec, graphs, device)
    pred["n_frag"] = [int(g.num_nodes) for g in graphs]
    pred["smiles"] = [g.smiles for g in graphs]
    return pred


def attention_maps(spec, graphs, device, n=6, out_path="figures/attention_maps.png", seed=0, min_frag=3):
    """Molecules drawn with the GIN's readout attention on each fragment (darker = more attention)."""
    import random
    from cam_mol import _caption, _images, _stack_rows, attention, draw_fragment_map
    from gspell_mol import canonical_graph
    pool = [g for g in graphs if g.num_nodes >= min_frag]
    random.Random(seed).shuffle(pool)
    tiles = []
    for g in pool[:n]:
        d = canonical_graph(g.cpu())
        a = attention(spec, d, device).numpy()
        png = draw_fragment_map(d.smiles, d.atom_groups, frag_scores=a - 1.0 / len(a), size=(320, 260))
        top = int(np.argmax(a))
        tiles.append(_caption(_images([png])[0],
                              [f"y = {float(d.y):.2f}   max attention F{top + 1} ({a[top]:.2f})"], size=13))
    rows = [tiles[i:i + 3] for i in range(0, len(tiles), 3)]
    grid = _stack_rows(rows, pad=12)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    grid.save(out_path)
    return grid


# ----------------------------------------------------------------------------- explanations

def explanation_heatmap(rec, path, title=""):
    """One molecule: fragments (rows) x explainers (columns). Each column is scaled to [-1, 1]
    within the molecule so methods with different units can be compared by rank and sign."""
    cols = [("GSPELL", None), ("Shapley (reference)", "shapley"), ("fragment Grad-CAM", "frag_gradcam"),
            ("atom Grad-CAM", "atom_gradcam"), ("readout attention", "attention"), ("GNNExplainer", "gnnexp")]
    cols = [(n, k) for n, k in cols if k is None or isinstance(rec.get(f"{k}_scores"), (list, tuple))]
    n = rec["n_fragments"]
    M = np.zeros((n, len(cols)))
    for j, (_, k) in enumerate(cols):
        if k is None:
            M[:, j] = [1.0 if i in rec["gspell_keep"] else 0.0 for i in range(n)]
        else:
            v = np.asarray(rec[f"{k}_scores"], dtype=float)
            if k in ("attention", "gnnexp"):
                v = v - v.mean()
            M[:, j] = v / max(np.abs(v).max(), 1e-8)
    flips = rec.get("removal_flips") if isinstance(rec.get("removal_flips"), (list, tuple)) else [False] * n
    ions = rec.get("counterion") if isinstance(rec.get("counterion"), (list, tuple)) else [False] * n
    labels = [f"F{i + 1} " + t.split(" (")[0][:22] + (" [ion]" if ions[i] else "") + (" (removal flips)" if flips[i] else "")
              for i, t in enumerate(rec["fragments"])]
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2.6, 0.38 * n + 1.3))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [c for c, _ in cols], rotation=30, ha="right")
    ax.set_yticks(range(n), labels, fontsize=8)
    for i in range(n):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:+.2f}" if j else ("YES" if M[i, j] else "no"), ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, label="scaled score (red supports prediction)")
    ax.set_title(title or rec["smiles"][:60], fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    return fig


@torch.no_grad()
def alignment_figure(spec, proj, graphs, text_emb_fn, device, title, path, n_frag=1500, seed=0):
    """UMAP (or t-SNE) of projected fragment soft prompts (mean over k tokens) together with the
    LLM text embeddings of the same fragments, the two sides GSPELL's context loss aligns."""
    from gspell_mol import collect_embeddings
    Hn, Tn, _, _ = collect_embeddings(spec.gin, graphs, device)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Tn), min(n_frag, len(Tn)), replace=False)
    Z = torch.nn.functional.normalize(proj(Hn[idx].to(device)).mean(1).float(), dim=-1).cpu().numpy()
    T = torch.nn.functional.normalize(text_emb_fn([Tn[i] for i in idx]).float(), dim=-1).numpy()
    cos = (Z * T).sum(1)
    method = available_methods()[-1]
    E = _reduce(np.concatenate([Z, T]), method, seed)
    groups = np.array([fragment_group(Tn[i]) for i in idx])
    common = [g for g, _ in Counter(groups).most_common(6)]
    cmap = matplotlib.colormaps["tab10"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax = axes[0]
    ax.scatter(*E[:len(idx)].T, s=5, color="#e07b39", alpha=0.6, label="soft prompts (projected GIN)", edgecolors="none")
    ax.scatter(*E[len(idx):].T, s=5, color="#4c78a8", alpha=0.6, label="LLM text embeddings", edgecolors="none")
    ax.legend(frameon=False, fontsize=7, markerscale=3); ax.set_title(f"{method.upper()}: both spaces", fontsize=9)
    ax = axes[1]
    for k, g in enumerate(common):
        sel = groups == g
        ax.scatter(*E[:len(idx)][sel].T, s=6, color=cmap(k), label=g, edgecolors="none")
    ax.set_title("soft prompts by functional group", fontsize=9)
    ax.legend(frameon=False, fontsize=7, markerscale=3, loc="center left", bbox_to_anchor=(1, 0.5))
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{title}  (mean cosine soft prompt vs. its text: {cos.mean():.3f})", fontsize=10)
    fig.tight_layout()
    _save(fig, path)
    return fig, float(cos.mean())


# ----------------------------------------------------------------------------- faithfulness figures

def faithfulness_figure(deletion_tables, agreement_tables, title, path):
    """Left: fragments removed (share) before the class flips, per explainer and dataset (lower =
    the ranking finds the decisive fragments sooner). Right: GSPELL vs. the text-only LLM on
    agreement with Shapley (AUROC: do YES fragments carry higher Shapley values?) and recall of
    the fragments whose removal alone flips the prediction."""
    import pandas as pd
    names = list(deletion_tables)
    methods = ["Shapley (reference)", "GSPELL (graph-level)", "LLM text-only (no projector)",
               "Grad-CAM, fragment GIN", "Grad-CAM, atom GIN -> fragments", "Readout attention",
               "GNNExplainer", "Random order"]
    methods = [m for m in methods if any(m in deletion_tables[n].index for n in names)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    h = 0.8 / max(1, len(names))
    cmap = matplotlib.colormaps["tab10"]
    y = np.arange(len(methods))[::-1]          # first method at the top
    for j, n in enumerate(names):
        t = deletion_tables[n]
        vals = [t.loc[m, "steps_to_flip"] if m in t.index else np.nan for m in methods]
        ax.barh(y + 0.4 - h / 2 - j * h, vals, h, label=n, color=cmap(j))
    ax.set_yticks(y, methods, fontsize=7.5)
    ax.set_xlabel("share of fragments removed before the class flips (lower = better)", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(names))
    ax = axes[1]
    rows = []
    for n in names:
        a = agreement_tables[n]
        for m in ["GSPELL (graph-level)", "LLM text-only (no projector)"]:
            if m in a.index:
                rows.append({"dataset": n, "method": "GSPELL" if m.startswith("GSPELL") else "text-only LLM",
                             "AUROC vs Shapley": a.loc[m].get("auroc_phi", np.nan),
                             "recall of decisive": a.loc[m].get("recall_decisive", np.nan)})
    df = pd.DataFrame(rows)
    if len(df):
        metrics = ["AUROC vs Shapley", "recall of decisive"]
        groups = [(d, m) for d in names for m in ["GSPELL", "text-only LLM"]]
        x = np.arange(len(names))
        for k, (meth, col) in enumerate([("GSPELL", "#e07b39"), ("text-only LLM", "#9aa5b1")]):
            for mi, metric in enumerate(metrics):
                vals = [df[(df.dataset == d) & (df.method == meth)][metric].mean() for d in names]
                ax.bar(x + (mi * 2 + k) * 0.2 - 0.3, vals, 0.18, color=col, alpha=1.0 if mi == 0 else 0.55,
                       label=f"{meth}: {metric}")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_xticks(x, names); ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_title("LLM support decisions vs. model references", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, path)
    return fig


def qa_figure(qa_tables, title, path):
    """QA accuracy with and without the GIN's soft prompts, per question type, against chance."""
    names = list(qa_tables)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.2), squeeze=False, sharey=True)
    for ax, n in zip(axes[0], names):
        t = qa_tables[n]
        t = t.drop(index="all", errors="ignore")
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["with_soft_prompts"], 0.38, color="#e07b39", label="with soft prompts")
        ax.bar(x + 0.2, t["text_only"], 0.38, color="#9aa5b1", label="text only")
        ax.scatter(x, t["chance"], marker="_", s=400, color="black", label="chance", zorder=3)
        ax.set_xticks(x, [i.replace("_", "\n") for i in t.index], fontsize=7)
        ax.set_ylim(0, 1.05); ax.set_title(n, fontsize=9)
    axes[0, 0].set_ylabel("accuracy")
    axes[0, -1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, path)
    return fig
