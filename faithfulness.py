"""Model-derived references for judging explanations: exact fragment Shapley values,
removal counterfactuals, deletion curves, agreement metrics, and a QA faithfulness test.

The game (Roth, JCIM 2026; computed on the fragment GIN)
  players  = the molecule's fragments (n <= 12, so all 2^n coalitions are enumerated)
  v(S)     = the GIN's output on the subgraph induced by the fragments in S, ORIENTED toward the
             predicted class and centred on the decision boundary:
                 v(S) = s * (raw(S) - c)
             raw = logit (classification) or standardized value (regression); c = 0 for logits,
             (threshold - mu) / sd for regression; s = +1 if the full molecule is predicted
             positive / above the threshold, else -1. So v(S) > 0 means "the subgraph S still
             gets the same class as the whole molecule", and phi_i > 0 means fragment i pushes
             toward the prediction.
  v(empty) = the prediction head applied to a zero molecule embedding (the GIN's attention
             readout of an empty set), i.e. the model's "no evidence" output.

Exactness is checked by the efficiency axiom: sum_i phi_i = v(all) - v(empty).

Removal counterfactual for fragment i: does v(all minus i) < 0, i.e. does removing i alone flip
the class? Taken from the same table, no extra model calls.

Deletion curve for any ranking (Shapley, Grad-CAM, attention, GNNExplainer, GSPELL's YES set):
remove fragments in ranked order and read v from the table. Reported as (a) the share of
fragments removed before the class flips (lower = the ranking found the decisive fragments
sooner) and (b) the area under v / v(all) (lower = faster drop). Random orderings give chance.

QA faithfulness: questions whose answers come from the model (Shapley, removal flips), asked to
the LLM with the GIN's soft prompts and without them. If accuracy is no better with soft
prompts, the LLM's rationale is driven by its chemistry priors, not by the model.
"""

import json
import os
import random
import re
from math import factorial

import numpy as np

# ----------------------------------------------------------------------------- the game


def popcounts(n):
    return np.array([bin(m).count("1") for m in range(2 ** n)])


def shapley_from_table(v, n):
    """Exact Shapley values from a table v[mask] over all 2^n coalitions (bit i = fragment i)."""
    v = np.asarray(v, dtype=float)
    masks = np.arange(2 ** n)
    size = popcounts(n)
    w = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
    phi = np.zeros(n)
    for i in range(n):
        bit = 1 << i
        without = masks[(masks & bit) == 0]
        phi[i] = np.sum(w[size[without]] * (v[without | bit] - v[without]))
    return phi


def full_mask(n):
    return (1 << n) - 1


def removal_flips(v, n):
    """For each fragment: does removing it alone flip the predicted class?"""
    full = full_mask(n)
    return [bool(v[full & ~(1 << i)] < 0) for i in range(n)]


def deletion_curve(v, n, order):
    """v after removing fragments in `order`, one at a time (length n + 1, starts at v(all))."""
    mask = full_mask(n)
    vals = [v[mask]]
    for i in order:
        mask &= ~(1 << int(i))
        vals.append(v[mask])
    return np.array(vals)


def deletion_metrics(v, n, order):
    curve = deletion_curve(v, n, order)
    flipped = np.where(curve[1:] < 0)[0]
    steps = (flipped[0] + 1) / n if len(flipped) else 1.0 + 1.0 / n   # never flips -> worst score
    base = curve[0] if curve[0] > 0 else 1.0
    auc = float(np.mean(curve / base))
    return {"steps_to_flip": float(steps), "deletion_auc": auc}


def random_deletion(v, n, n_perm=50, seed=0):
    rng = random.Random(seed)
    res = [deletion_metrics(v, n, rng.sample(range(n), n)) for _ in range(n_perm)]
    return {k: float(np.mean([r[k] for r in res])) for k in res[0]}


def ranking_from_set(keep, n, seed=0, n_perm=20):
    """Orders that put a set's members first (ties broken at random), for set-valued explainers."""
    rng = random.Random(seed)
    keep = list(keep)
    rest = [i for i in range(n) if i not in set(keep)]
    out = []
    for _ in range(n_perm):
        a, b = keep[:], rest[:]
        rng.shuffle(a); rng.shuffle(b)
        out.append(a + b)
    return out


# ----------------------------------------------------------------------------- the model side

def coalition_table(spec, data, device, batch_size=1024):
    """Oriented, centred values v[mask] for all 2^n coalitions of a fragment graph."""
    import torch
    from torch_geometric.data import Batch, Data
    from torch_geometric.utils import subgraph

    gin = spec.gin.eval()
    n = int(data.num_nodes)
    raw = np.zeros(2 ** n)
    with torch.no_grad():
        hidden = gin.head[0].in_features
        raw[0] = gin.head(torch.zeros(1, hidden, device=device)).view(-1).item()
        masks = list(range(1, 2 ** n))
        x, ei, ea = data.x.cpu(), data.edge_index.cpu(), data.edge_attr.cpu()
        for start in range(0, len(masks), batch_size):
            chunk = masks[start:start + batch_size]
            graphs = []
            for m in chunk:
                keep = torch.tensor([i for i in range(n) if m >> i & 1], dtype=torch.long)
                sei, sea = subgraph(keep, ei, ea, relabel_nodes=True, num_nodes=n)
                graphs.append(Data(x=x[keep], edge_index=sei, edge_attr=sea))
            b = Batch.from_data_list(graphs).to(device)
            raw[chunk] = gin(b).view(-1).cpu().numpy()
    if spec.task == "classification":
        c = 0.0
    else:
        c = (spec.threshold - spec.mu) / spec.sd
    s = 1.0 if raw[full_mask(n)] >= c else -1.0
    return s * (raw - c), s


def analyse_molecule(spec, data, device, rec=None, n_perm=50):
    """Shapley values, removal flips, and deletion metrics for one molecule; when `rec` (a row of
    the explanation records) is given, also scores every explainer in it against Shapley."""
    n = int(data.num_nodes)
    v, sign = coalition_table(spec, data, device)
    phi = shapley_from_table(v, n)
    out = {"smiles": data.smiles, "n_fragments": n, "v_full": float(v[full_mask(n)]), "v_empty": float(v[0]),
           "shapley_scores": phi.tolist(), "removal_flips": removal_flips(v, n),
           "efficiency_error": float(abs(phi.sum() - (v[full_mask(n)] - v[0]))),
           "pred_sign": sign}
    out["deletion"] = {"shapley": deletion_metrics(v, n, np.argsort(-phi)), "random": random_deletion(v, n, n_perm)}
    if rec is not None:
        out["agreement"] = agreement_with_shapley(rec, phi, out["removal_flips"])
        for key in SCORE_EXPLAINERS:
            if isinstance(rec.get(f"{key}_scores"), (list, tuple)):
                out["deletion"][key] = deletion_metrics(v, n, np.argsort(-np.asarray(rec[f"{key}_scores"])))
        for key in SET_EXPLAINERS:
            if isinstance(rec.get(f"{key}_keep"), (list, tuple)):
                res = [deletion_metrics(v, n, o) for o in ranking_from_set(rec[f"{key}_keep"], n)]
                out["deletion"][key] = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    return out


# ----------------------------------------------------------------------------- agreement

SCORE_EXPLAINERS = ["frag_gradcam", "atom_gradcam", "attention", "gnnexp"]
SET_EXPLAINERS = ["gspell", "llm"]


def _auroc(pos_scores, neg_scores):
    """P(random positive > random negative), ties count half."""
    if not len(pos_scores) or not len(neg_scores):
        return np.nan
    p, q = np.asarray(pos_scores)[:, None], np.asarray(neg_scores)[None, :]
    return float(((p > q) + 0.5 * (p == q)).mean())


def agreement_with_shapley(rec, phi, flips):
    """How well each explainer matches the model-derived references for one molecule."""
    from scipy.stats import spearmanr
    n = len(phi)
    phi = np.asarray(phi)
    support = {i for i in range(n) if phi[i] > 0}          # fragments pushing toward the prediction
    decisive = {i for i in range(n) if flips[i]}           # removing them alone flips the class
    res = {}
    for key in SET_EXPLAINERS:
        if not isinstance(rec.get(f"{key}_keep"), (list, tuple)):
            continue
        yes = set(rec[f"{key}_keep"])
        no = set(range(n)) - yes
        k = max(1, len(yes))
        topk = set(np.argsort(-phi)[:k].tolist())
        res[key] = {
            "auroc_phi": _auroc(phi[list(yes)], phi[list(no)]),           # do YES fragments have higher phi?
            "precision_support": len(yes & support) / len(yes) if yes else np.nan,
            "recall_support": len(yes & support) / len(support) if support else np.nan,
            "jaccard_topk": len(yes & topk) / len(yes | topk) if yes else 0.0,
            "recall_decisive": len(yes & decisive) / len(decisive) if decisive else np.nan,
            "size": len(yes),
        }
    for key in SCORE_EXPLAINERS:
        if not isinstance(rec.get(f"{key}_scores"), (list, tuple)):
            continue
        sc = np.asarray(rec[f"{key}_scores"], dtype=float)
        rho = spearmanr(sc, phi).correlation if n >= 3 and np.ptp(sc) > 1e-9 and np.ptp(phi) > 1e-9 else np.nan
        top1 = int(np.argmax(sc)) == int(np.argmax(phi))
        res[key] = {"spearman_phi": float(rho) if rho == rho else np.nan, "top1_match": float(top1),
                    "recall_decisive": (len(set(np.argsort(-sc)[:max(1, len(decisive))].tolist()) & decisive)
                                        / len(decisive)) if decisive else np.nan}
    return res


# ----------------------------------------------------------------------------- QA faithfulness

def make_questions(phi, flips, seed=0):
    """Questions whose correct answers come from the model. Fragment numbers are 1-based."""
    rng = random.Random(seed)
    n = len(phi)
    phi = np.asarray(phi)
    qs = []
    if n >= 2 and len(set(np.round(phi, 9))) > 1:
        qs.append({"type": "most_important", "answer": int(np.argmax(phi)) + 1, "chance": 1 / n,
                   "text": "Which single fragment contributes most to the GNN's prediction? "
                           "Answer with the fragment number only."})
    if n >= 2 and phi.min() < 0:
        qs.append({"type": "most_opposing", "answer": int(np.argmin(phi)) + 1, "chance": 1 / n,
                   "text": "Which single fragment works most against the GNN's prediction? "
                           "Answer with the fragment number only."})
    yes = [i for i in range(n) if flips[i]]
    no = [i for i in range(n) if not flips[i]]
    for pool, ans in ((yes, "YES"), (no, "NO")):            # one flipping and one non-flipping, when present
        if pool:
            i = rng.choice(pool)
            qs.append({"type": "removal_flip", "answer": ans, "chance": 0.5, "fragment": i + 1,
                       "text": f"If fragment {i + 1} were removed from the molecule, would the GNN's "
                               f"prediction change? Answer YES or NO only."})
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if abs(phi[a] - phi[b]) > 1e-6]
    if pairs:
        a, b = rng.choice(pairs)
        if rng.random() < 0.5:
            a, b = b, a
        qs.append({"type": "pairwise", "answer": (a if phi[a] > phi[b] else b) + 1, "chance": 0.5,
                   "options": [a + 1, b + 1],
                   "text": f"Which fragment matters more for the GNN's prediction: fragment {a + 1} or "
                           f"fragment {b + 1}? Answer with the fragment number only."})
    return qs


def qa_prompt(data, label_text, question, with_soft=True):
    z = lambda key: f" <<Z_{key}>>" if with_soft else ""
    lines = [f"Target molecule:{z('G')}",
             f"GNN prediction: the molecule {label_text}.",
             "Fragments of the molecule:"]
    for i, t in enumerate(data.fragment_text):
        lines.append(f"Fragment {i + 1}: {t}{z(i)}")
    lines += ["", "Answer this question about the GNN's own reasoning, using the embeddings shown:", question]
    return "\n".join(lines)


def parse_answer(text, qtype, n, options=None):
    if qtype == "removal_flip":
        m = re.search(r"\b(YES|NO)\b", text, flags=re.I)
        return m.group(1).upper() if m else None
    for m in re.finditer(r"\d+", text):
        k = int(m.group())
        if 1 <= k <= n and (options is None or k in options):
            return k
    return None


def ask_questions(spec, proj, tok, llm, data, label_text, device, phi, flips, seed=0, max_new_tokens=12):
    """Ask each question with and without soft prompts. Returns a list of answered questions."""
    import torch
    from gspell_mol import generate
    n = int(data.num_nodes)
    with torch.no_grad():
        d = data.clone().to(device)
        d.batch = torch.zeros(n, dtype=torch.long, device=device)
        h, g = spec.gin.embed(d)
        zs = proj(torch.cat([h, g]))
    soft = {str(i): zs[i] for i in range(n)}
    soft["G"] = zs[n]
    out = []
    for q in make_questions(phi, flips, seed):
        row = dict(q)
        for cond, use_soft in (("soft", True), ("text_only", False)):
            raw = generate(tok, llm, qa_prompt(data, label_text, q["text"], use_soft),
                           soft if use_soft else None, max_new_tokens=max_new_tokens)
            ans = parse_answer(raw, q["type"], n, q.get("options"))
            row[f"{cond}_raw"] = raw.strip()[:80]
            row[f"{cond}_answer"] = ans
            row[f"{cond}_correct"] = float(ans == q["answer"])
        out.append(row)
    return out


# ----------------------------------------------------------------------------- summaries

METHOD_LABELS = {"gspell": "GSPELL (graph-level)", "llm": "LLM text-only (no projector)",
                 "frag_gradcam": "Grad-CAM, fragment GIN", "atom_gradcam": "Grad-CAM, atom GIN -> fragments",
                 "attention": "Readout attention", "gnnexp": "GNNExplainer", "shapley": "Shapley (reference)",
                 "random": "Random order"}


def summarize_agreement(rows):
    """rows: analyse_molecule outputs. Mean of each agreement metric per explainer."""
    import pandas as pd
    acc = {}
    for r in rows:
        for key, metrics in r.get("agreement", {}).items():
            for m, val in metrics.items():
                acc.setdefault((key, m), []).append(val)
    table = {}
    for (key, m), vals in acc.items():
        vals = [x for x in vals if x == x]          # drop NaN (metric undefined for that molecule)
        table.setdefault(METHOD_LABELS.get(key, key), {})[m] = float(np.mean(vals)) if vals else np.nan
    return pd.DataFrame(table).T


def summarize_deletion(rows):
    import pandas as pd
    acc = {}
    for r in rows:
        for key, metrics in r["deletion"].items():
            for m, val in metrics.items():
                acc.setdefault((key, m), []).append(val)
    table = {}
    for (key, m), vals in acc.items():
        table.setdefault(METHOD_LABELS.get(key, key), {})[m] = float(np.mean(vals))
    df = pd.DataFrame(table).T
    return df.sort_values("steps_to_flip") if "steps_to_flip" in df else df


def summarize_qa(qa_rows):
    import pandas as pd
    df = pd.DataFrame(qa_rows)
    if not len(df):
        return df
    g = df.groupby("type").agg(questions=("type", "size"), with_soft_prompts=("soft_correct", "mean"),
                               text_only=("text_only_correct", "mean"), chance=("chance", "mean"))
    total = pd.DataFrame([{"questions": len(df), "with_soft_prompts": df["soft_correct"].mean(),
                           "text_only": df["text_only_correct"].mean(), "chance": df["chance"].mean()}], index=["all"])
    g = pd.concat([g, total])
    g["grounding_gain"] = g["with_soft_prompts"] - g["text_only"]
    return g


def load_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")
