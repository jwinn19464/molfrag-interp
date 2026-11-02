"""GSPELL adapted to graph-level explanation on fragment graphs.

Original method (Baghershahi et al., arXiv:2508.07117) explains NODE predictions on
text-attributed graphs. Here the target is the MOLECULE: its graph embedding is the
target soft prompt, and its BRICS fragments play the role of the neighbors whose
support the LLM judges.

Steps
  1. Frozen fragment-GIN gives fragment embeddings h_v and molecule embeddings h_G.
  2. A projector maps each embedding to k soft-prompt tokens in the LLM's embedding
     space, trained with GSPELL's context-alignment + contrastive losses.
  3. A hybrid prompt interleaves soft prompts with fragment text; the frozen LLM
     returns YES/NO support per fragment. YES fragments form the explanation subgraph.
  4. Fidelity (does the GIN give the same prediction on the explanation subgraph?)
     and size are compared with GNNExplainer, a random baseline, and a text-only LLM.
"""

import json
import math
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph

TASK_TEXT = {
    "BBBP": ("penetrates the blood-brain barrier", "does not penetrate the blood-brain barrier"),
    "BACE": ("inhibits BACE-1", "does not inhibit BACE-1"),
    # Regression: the prediction is turned into a class at the training-set median
    "ESOL": ("is more water-soluble than the median compound in the dataset",
             "is less water-soluble than the median compound in the dataset"),
}

SYSTEM_PROMPT = (
    "You are a medicinal chemist explaining the prediction of a graph neural network (GNN). "
    "The GNN represents a molecule as a graph of chemical fragments. Each fragment is shown with "
    "its SMILES, its functional groups, and the GNN's internal embedding of it."
)


# ----------------------------------------------------------------------------- LLM

def load_llm(model_id="meta-llama/Llama-3.1-8B-Instruct", low_memory=None):
    """bf16 on GPUs with >= 22 GB and bf16 support (L4, A100); 4-bit fp16 otherwise (T4)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if low_memory is None:
        cap = torch.cuda.get_device_capability()
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        low_memory = cap[0] < 8 or mem < 22
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if low_memory:
        llm = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16))
    else:
        llm = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    llm.eval()
    return tok, llm


@torch.no_grad()
def text_embeddings(texts, tok, llm, batch_size=256):
    """LLM(v) in GSPELL's context loss: mean of the LLM's input token embeddings of the text."""
    emb = llm.get_input_embeddings()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        ids = enc.input_ids.to(emb.weight.device)
        mask = enc.attention_mask.to(emb.weight.device).unsqueeze(-1).float()
        e = emb(ids).float()
        out.append(((e * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu())
    return torch.cat(out)


# ----------------------------------------------------------------------------- projector

class Projector(nn.Module):
    """GNN embedding -> k soft tokens. Outputs are rescaled to the LLM's average
    token-embedding norm so the soft prompts sit in the same range as real tokens."""

    def __init__(self, in_dim, llm_dim, k=5, hidden=512, token_norm=1.0):
        super().__init__()
        self.k, self.llm_dim = k, llm_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, k * llm_dim))
        self.register_buffer("token_norm", torch.tensor(float(token_norm)))

    def forward(self, h):
        z = self.net(h).view(-1, self.k, self.llm_dim)
        return F.normalize(z, dim=-1) * self.token_norm


@torch.no_grad()
def collect_embeddings(gin, graphs, device, batch_size=128):
    """Node embeddings + fragment text, and graph embeddings + molecule text."""
    gin.eval()
    H_nodes, T_nodes, H_graphs, T_graphs = [], [], [], []
    for b in DataLoader(graphs, batch_size=batch_size):
        h, g = gin.embed(b.to(device))
        H_nodes.append(h.cpu()); H_graphs.append(g.cpu())
        T_nodes += [t for texts in b.fragment_text for t in texts]
        T_graphs += [f"molecule {s}" for s in b.smiles]
    return torch.cat(H_nodes), T_nodes, torch.cat(H_graphs), T_graphs


def train_projector(H, T, llm_dim, token_norm, device, k=5, epochs=30, batch_size=512,
                    tau=0.2, w_context=1.0, w_contrast=0.5, lr=1e-3, seed=0):
    """GSPELL losses: L = w_context * L_context + w_contrast * L_contrast."""
    torch.manual_seed(seed)
    proj = Projector(H.size(1), llm_dim, k=k, token_norm=token_norm).to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)
    H, T = H.to(device), F.normalize(T.to(device), dim=-1)
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(H.size(0), device=device)
        total = 0.0
        for i in range(0, H.size(0), batch_size):
            idx = perm[i:i + batch_size]
            zbar = F.normalize(proj(H[idx]).mean(1), dim=-1)
            l_ctx = -(zbar * T[idx]).sum(-1).mean()
            hn = F.normalize(H[idx], dim=-1)
            p_gnn = F.softmax(hn @ hn.T / tau, dim=-1)
            log_p_proj = F.log_softmax(zbar @ zbar.T / tau, dim=-1)
            l_con = -(p_gnn * log_p_proj).sum(-1).mean()
            loss = w_context * l_ctx + w_contrast * l_con
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        history.append(total / H.size(0))
    proj.eval()
    return proj, history


# ----------------------------------------------------------------------------- prompting

def build_user_prompt(data, label_text, with_soft=True):
    z = lambda key: f" <<Z_{key}>>" if with_soft else ""
    lines = [f"Target molecule:{z('G')}",
             f"GNN prediction: the molecule {label_text}.",
             "Fragments of the molecule:"]
    for i, t in enumerate(data.fragment_text):
        lines.append(f"Fragment {i + 1}: {t}{z(i)}")
    lines += ["",
              "For each fragment, write exactly one line in this format:",
              "Fragment <number>: Summary: <one sentence>. Support: YES or NO",
              f"YES means the fragment supports the prediction that the molecule {label_text}.",
              "Only use the fragment numbers listed above. After the fragments, write one line "
              "starting with 'Reasoning:' that explains the prediction in one or two sentences."]
    return "\n".join(lines)


@torch.no_grad()
def generate(tok, llm, user_prompt, soft=None, max_new_tokens=400):
    """Chat-formatted prompt with <<Z_key>> markers replaced by soft-prompt tensors."""
    emb = llm.get_input_embeddings()
    dev, dtype = emb.weight.device, emb.weight.dtype
    text = tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": user_prompt}],
                                   tokenize=False, add_generation_prompt=True)
    pieces = []
    for part in re.split(r"(<<Z_\w+>>)", text):
        m = re.fullmatch(r"<<Z_(\w+)>>", part)
        if m:
            pieces.append(soft[m.group(1)].to(dev, dtype))
        elif part:
            ids = tok(part, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
            pieces.append(emb(ids)[0])
    inputs = torch.cat(pieces).unsqueeze(0)
    out = llm.generate(inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2], device=dev, dtype=torch.long),
                       max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def parse_support(text, n_frag):
    """Map fragment index -> True (YES) / False (NO). Out-of-range numbers are dropped
    (GSPELL's post-hoc hallucination filter) and counted."""
    support, hallucinated = {}, 0
    for line in text.splitlines():
        m = re.match(r"\W*Fragment\s*(\d+)\b(.*)", line, flags=re.I)
        if not m:
            continue
        s = re.search(r"Support\W*\s*(YES|NO)\b", m.group(2), flags=re.I)
        if not s:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < n_frag:
            support[idx] = s.group(1).upper() == "YES"
        else:
            hallucinated += 1
    return support, hallucinated


# ----------------------------------------------------------------------------- evaluation

class ModelSpec:
    """A trained fragment-GIN plus what's needed to turn its output into a class:
    probability >= 0.5 for classification, value >= training median for regression."""

    def __init__(self, gin, task, mu=0.0, sd=1.0, threshold=0.5):
        self.gin, self.task, self.mu, self.sd, self.threshold = gin, task, mu, sd, threshold

    def score(self, out):
        return torch.sigmoid(out) if self.task == "classification" else out * self.sd + self.mu


def load_spec(dataset, seed, device, graph_dir="frag_graphs", ckpt_dir="checkpoints"):
    from fragment_graphs import DATASETS
    from frag_gin import load_checkpoint, load_graphs
    gin, ckpt = load_checkpoint(f"{ckpt_dir}/{dataset}_gin_seed{seed}.pt", device)
    task = DATASETS[dataset][1]
    threshold = 0.5
    if task == "regression":
        train = load_graphs(dataset, graph_dir)[0]
        threshold = float(torch.cat([d.y for d in train]).median())
    return ModelSpec(gin, task, ckpt["mu"], ckpt["sd"], threshold)


def canonical_graph(data):
    """Rebuild a molecule's fragment graph from its stored (canonical) SMILES so that
    `atom_groups` index the atoms of Chem.MolFromSmiles(data.smiles). Needed to map
    fragment scores onto atoms and to align with atom-level models."""
    from fragment_graphs import fragment_graph, to_pyg
    g = fragment_graph(data.smiles, ionic=bool(getattr(data, "ionic", False)))
    g["smiles"] = data.smiles     # atom_groups index the atoms of exactly this string
    return to_pyg(g, float(data.y), None)


@torch.no_grad()
def predict(spec, data, device, keep=None):
    """Model score (probability or value) on the full graph or on the subgraph of `keep` fragments."""
    if keep is None:
        d = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    else:
        keep_t = torch.tensor(sorted(keep), dtype=torch.long)
        ei, ea = subgraph(keep_t, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=data.num_nodes)
        d = Data(x=data.x[keep_t], edge_index=ei, edge_attr=ea)
    d.batch = torch.zeros(d.x.size(0), dtype=torch.long)
    spec.gin.eval()   # BatchNorm must use running stats: explanation subgraphs can have one node
    return spec.score(spec.gin(d.to(device))).item()


def fidelity(spec, data, keep, full_score, device):
    if not keep:
        return 0.0        # an empty explanation cannot reproduce a prediction
    return float((predict(spec, data, device, keep) >= spec.threshold) == (full_score >= spec.threshold))


class _GraphWrap(nn.Module):
    def __init__(self, gin):
        super().__init__()
        self.gin = gin

    def forward(self, x, edge_index, edge_attr, batch):
        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        d.batch = batch
        return self.gin(d).view(-1, 1)


def gnnexplainer_mask(spec, data, device, epochs=200):
    from torch_geometric.explain import Explainer, GNNExplainer
    mode = "binary_classification" if spec.task == "classification" else "regression"
    explainer = Explainer(
        model=_GraphWrap(spec.gin).eval(), algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model", node_mask_type="object", edge_mask_type=None,
        model_config=dict(mode=mode, task_level="graph", return_type="raw"))
    d = data.clone().to(device)
    batch = torch.zeros(d.num_nodes, dtype=torch.long, device=device)
    exp = explainer(d.x, d.edge_index, edge_attr=d.edge_attr, batch=batch)
    spec.gin.eval()   # the Explainer restores the wrapper's training flag afterwards
    return exp.node_mask.view(-1).detach().cpu()


def top_k(scores, k):
    return torch.as_tensor(scores).argsort(descending=True)[:k].tolist()


def _done_smiles(path):
    import os
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_explanations(spec, proj, tok, llm, graphs, dataset, device, n_eval=30, min_frag=2,
                     max_frag=12, text_only_baseline=True, extra_explainers=None, seed=0,
                     log_every=5, records_path=None, only_counterions=False, indices=None):
    """Explain n_eval test molecules.

    extra_explainers: {name: fn(spec, data, device) -> per-fragment scores}. Each is
    evaluated at GSPELL's explanation size (top-k). Scores are stored for agreement analysis.
    only_counterions: ionic mode; explain only molecules that have counter-ion nodes.
    indices: explain exactly graphs[i] for these i (used to explain the SAME molecules in
    both fragmentation modes); overrides n_eval, min/max_frag and only_counterions.
    records_path: a .jsonl file; finished molecules are appended as they complete, and
    molecules already in the file are skipped, so an interrupted run can be resumed.
    Returns (per-molecule records, summary table)."""
    pos_text, neg_text = TASK_TEXT[dataset]
    extra_explainers = extra_explainers or {}
    rng = random.Random(seed)
    if indices is None:
        pool = [i for i, d in enumerate(graphs) if min_frag <= d.num_nodes <= max_frag]
        if only_counterions:
            pool = [i for i in pool if hasattr(graphs[i], "counterion") and bool(graphs[i].counterion.any())]
        random.Random(seed).shuffle(pool)  # fixed order: raising n_eval later extends the same list
        indices = pool[:n_eval]
    chosen = [graphs[i] for i in indices]
    wanted = {d.smiles for d in chosen}
    index_of = {d.smiles: int(i) for i, d in zip(indices, chosen)}
    records = _done_smiles(records_path)
    for r in records:                 # entries written by older versions lack these fields
        r.setdefault("test_index", index_of.get(r["smiles"]))
        r.setdefault("counterion", [False] * r["n_fragments"])
    done = {r["smiles"] for r in records}
    if done:
        print(f"  resuming: {len(done)} molecules already explained")
    for j, (test_index, data) in enumerate(zip(indices, chosen)):
        if data.smiles in done:
            continue
        data = canonical_graph(data.cpu())
        n = data.num_nodes
        full = predict(spec, data, device)
        label_text = pos_text if full >= spec.threshold else neg_text
        rec = {"smiles": data.smiles, "test_index": int(test_index), "n_fragments": n, "gin_score": full, "y": float(data.y),
               "fragments": list(data.fragment_text), "atom_groups": data.atom_groups,
               "counterion": data.counterion.tolist() if hasattr(data, "counterion") else [False] * n}

        # GSPELL (soft prompts)
        with torch.no_grad():
            d = data.clone().to(device)
            d.batch = torch.zeros(n, dtype=torch.long, device=device)
            h, g = spec.gin.embed(d)
            zs = proj(torch.cat([h, g]))
        soft = {str(i): zs[i] for i in range(n)}
        soft["G"] = zs[n]
        out = generate(tok, llm, build_user_prompt(data, label_text, True), soft)
        sup, hal = parse_support(out, n)
        keep = sorted(i for i, v in sup.items() if v)
        k = max(1, len(keep))
        rec.update(gspell_output=out, gspell_keep=keep, gspell_hallucinated=hal,
                   gspell_fid=fidelity(spec, data, keep, full, device), gspell_size=len(keep))

        # Text-only LLM (no projector)
        if text_only_baseline:
            out_t = generate(tok, llm, build_user_prompt(data, label_text, False))
            sup_t, _ = parse_support(out_t, n)
            keep_t = sorted(i for i, v in sup_t.items() if v)
            rec.update(llm_output=out_t, llm_keep=keep_t,
                       llm_fid=fidelity(spec, data, keep_t, full, device), llm_size=len(keep_t))

        # GNNExplainer: its own threshold, and size-matched to GSPELL
        mask = gnnexplainer_mask(spec, data, device)
        keep_g = [i for i in range(n) if mask[i] > 0.5] or [int(mask.argmax())]
        keep_gk = top_k(mask, k)
        rec.update(gnnexp_scores=mask.tolist(), gnnexp_keep=keep_g,
                   gnnexp_fid=fidelity(spec, data, keep_g, full, device), gnnexp_size=len(keep_g),
                   gnnexp_k_keep=keep_gk, gnnexp_k_fid=fidelity(spec, data, keep_gk, full, device),
                   gnnexp_k_size=k)

        # Gradient / attention explainers, size-matched to GSPELL
        for name, fn in extra_explainers.items():
            scores = torch.as_tensor(fn(spec, data, device), dtype=torch.float).view(-1)
            keep_e = top_k(scores, k)
            rec.update({f"{name}_scores": scores.tolist(), f"{name}_keep": keep_e,
                        f"{name}_fid": fidelity(spec, data, keep_e, full, device), f"{name}_size": k})

        # Random: half the fragments, 5 draws
        r_f = [fidelity(spec, data, rng.sample(range(n), math.ceil(n / 2)), full, device) for _ in range(5)]
        rec.update(random_fid=float(np.mean(r_f)), random_size=math.ceil(n / 2))

        records.append(rec)
        if records_path:
            with open(records_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        if (j + 1) % log_every == 0:
            print(f"  explained {j + 1}/{len(chosen)}")

    df = pd.DataFrame([r for r in records if r["smiles"] in wanted])   # ignore stale entries in the file
    return df, summarize(df, extra_explainers)


METHOD_NAMES = {"gspell": "GSPELL (graph-level)", "llm": "LLM text-only (no projector)",
                "gnnexp": "GNNExplainer", "gnnexp_k": "GNNExplainer (size-matched)",
                "frag_gradcam": "Grad-CAM, fragment GIN (size-matched)",
                "attention": "Readout attention (size-matched)",
                "atom_gradcam": "Grad-CAM, atom GIN -> fragments (size-matched)",
                "random": "Random (half)"}


def paired_indices(graph_lists, n_eval, seed=0, min_frag=2, max_frag=12, require_counterion=None):
    """Test-set positions that are explainable in every mode, in a fixed shuffled order.
    graph_lists: {mode: list of test graphs}; all modes come from the same CSV and scaffold
    split, so position i is the same molecule in each. require_counterion: a mode name whose
    graphs must contain a counter-ion node (to focus on salts)."""
    lists = list(graph_lists.values())
    n = len(lists[0])
    if any(len(g) != n for g in lists):
        raise ValueError("modes have different test-set sizes; rebuild both from the same CSVs")
    ok = [i for i in range(n) if all(min_frag <= g[i].num_nodes <= max_frag for g in lists)]
    if require_counterion:
        ok = [i for i in ok if bool(graph_lists[require_counterion][i].counterion.any())]
    random.Random(seed).shuffle(ok)
    return ok[:n_eval]


def summarize(df, extra_explainers=None):
    keys = ["gspell", "llm", "gnnexp", "gnnexp_k"] + list(extra_explainers or {}) + ["random"]
    return pd.DataFrame([
        {"method": METHOD_NAMES.get(key, key), "fidelity_%": 100 * df[f"{key}_fid"].mean(),
         "fidelity_std": 100 * df[f"{key}_fid"].std(), "size": df[f"{key}_size"].mean(),
         "size_frac_%": 100 * (df[f"{key}_size"] / df["n_fragments"]).mean(), "n": len(df)}
        for key in keys if f"{key}_fid" in df])


def save_records(df, path):
    with open(path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=1, default=str)
