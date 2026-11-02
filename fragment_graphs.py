"""
BRICS fragment graphs for MoleculeNet (BBBP, BACE, ESOL), scaffold splits.

Each molecule becomes a graph whose NODES are BRICS fragments and whose EDGES are
the bonds BRICS broke. Every node carries text (fragment SMILES + functional-group
names), so each molecule is a small text-attributed graph that GSPELL can explain.

Two modes
    default   counter-ions and other extra components are removed (largest component kept)
    --ionic   BRICS + ionic decomposition: every component is kept; metal-heteroatom bonds
              written as covalent (e.g. CC(=O)O[Na]) are split into ions; each counter-ion /
              salt partner is one node, linked to the main molecule by an "ionic" edge

Usage
    python fragment_graphs.py --demo "O=C(Nc1ccccc1)c1ccccc1"   # inspect one molecule
    python fragment_graphs.py --demo "CCN(CC)CC.Cl" --ionic     # same, keeping the salt
    python fragment_graphs.py --stats                           # dataset statistics table (CPU, minutes)
    python fragment_graphs.py --build                           # also save PyG graphs to ./frag_graphs/

Requires: rdkit, numpy, pandas, deepchem (for scaffold splits); torch + torch_geometric only for --build.
"""

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

DATASETS = {
    # name: (DeepChem loader, task type)
    "BBBP": ("load_bbbp", "classification"),
    "BACE": ("load_bace_classification", "classification"),
    "ESOL": ("load_delaney", "regression"),
}
MAX_FRAGMENTS_FOR_EXACT_SHAPLEY = 12
FP_BITS = 512

# Readable functional-group names for node text. A group is assigned to a fragment
# only if every atom of the match lies inside that fragment.
FUNCTIONAL_GROUPS = {
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "carboxylate": "[CX3](=O)[O-]",
    "protonated amine (ammonium)": "[NX4+;!$(N~O)]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "amide": "[NX3][CX3]=[OX1]",
    "ketone or aldehyde": "[#6][CX3](=O)[#1,#6]",
    "hydroxyl": "[OX2H][#6]",
    "ether": "[OD2]([#6])[#6]",
    "primary amine": "[NX3;H2;!$(NC=O)][#6]",
    "secondary amine": "[NX3;H1;!$(NC=O)]([#6])[#6]",
    "tertiary amine": "[NX3;H0;!$(NC=O);!$(N=*);!a]([#6])([#6])[#6]",
    "nitro": "[N+](=O)[O-]",
    "nitrile": "C#N",
    "sulfonamide": "S(=O)(=O)N",
    "sulfone or sulfonyl": "[SX4](=O)(=O)",
    "thioether": "[SX2]([#6])[#6]",
    "trifluoromethyl": "C(F)(F)F",
    "halogen": "[F,Cl,Br,I]",
    "phosphate or phosphonate": "P(=O)(O)",
    "benzene ring": "c1ccccc1",
    "aromatic nitrogen heterocycle": "[n]",
    "aromatic oxygen or sulfur heterocycle": "[o,s]",
    "alkene": "[CX3]=[CX3]",
    "carbonyl": "[CX3]=[OX1]",
}
# Drop generic labels when a more specific one already covers them.
SUPPRESS = {
    "carboxylic acid": {"hydroxyl", "carbonyl", "ketone or aldehyde"},
    "carboxylate": {"carbonyl", "ketone or aldehyde"},
    "ester": {"ether", "carbonyl", "ketone or aldehyde"},
    "amide": {"carbonyl", "ketone or aldehyde"},
    "ketone or aldehyde": {"carbonyl"},
    "sulfonamide": {"sulfone or sulfonyl"},
    "trifluoromethyl": {"halogen"},
}
ELEMENT_NAMES = {"N": "nitrogen", "O": "oxygen", "S": "sulfur", "C": "carbon", "P": "phosphorus"}
_FG_PATTERNS = {name: Chem.MolFromSmarts(s) for name, s in FUNCTIONAL_GROUPS.items()}
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
_LARGEST = rdMolStandardize.LargestFragmentChooser()
_METALS = rdMolStandardize.MetalDisconnector()

# Readable names for common counter-ions and salt partners (canonical SMILES of the component).
ION_NAMES = {
    "[Cl-]": "chloride ion", "Cl": "chloride (HCl salt)", "[Br-]": "bromide ion", "Br": "bromide (HBr salt)",
    "[I-]": "iodide ion", "I": "iodide (HI salt)", "[F-]": "fluoride ion",
    "[Na+]": "sodium ion", "[K+]": "potassium ion", "[Li+]": "lithium ion", "[Ca+2]": "calcium ion",
    "[Mg+2]": "magnesium ion", "[Zn+2]": "zinc ion", "[NH4+]": "ammonium ion", "N": "ammonia",
    "O": "water (hydrate)",
    "O=S(=O)(O)O": "sulfuric acid (sulfate salt)", "O=S(=O)([O-])[O-]": "sulfate ion",
    "O=S(=O)([O-])O": "hydrogen sulfate ion",
    "CS(=O)(=O)O": "methanesulfonic acid (mesylate salt)", "CS(=O)(=O)[O-]": "mesylate ion",
    "Cc1ccc(S(=O)(=O)O)cc1": "p-toluenesulfonic acid (tosylate salt)",
    "O=P(O)(O)O": "phosphoric acid (phosphate salt)", "O=[N+]([O-])O": "nitric acid (nitrate salt)",
    "O=C(O)/C=C\\C(=O)O": "maleic acid (maleate salt)", "O=C(O)/C=C/C(=O)O": "fumaric acid (fumarate salt)",
    "O=C(O)C(=O)O": "oxalic acid (oxalate salt)", "O=C(O)CC(O)(CC(=O)O)C(=O)O": "citric acid (citrate salt)",
    "O=C(O)C(O)C(O)C(=O)O": "tartaric acid (tartrate salt)", "CC(=O)O": "acetic acid (acetate salt)",
}
# Basic nitrogens that an acid counter-ion would protonate (amines and aromatic N, not amides)
_BASIC_N = Chem.MolFromSmarts("[NX3;!$(N-C=[O,S,N]);!$(N-S(=O)=O);!$(N-a)]")
_AROM_N = Chem.MolFromSmarts("[nX2]")
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


# ----------------------------------------------------------------------------- fragmentation

def prepare_mol(smiles, ionic=False):
    """Parse SMILES. Default: keep the largest component (drops counter-ions such as '.Cl').
    ionic=True: keep every component and split metal-heteroatom bonds into ions
    (RDKit MetalDisconnector; atom order is preserved).
    Returns (mol, changed) or (None, False) if the SMILES is invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, False
    if ionic:
        n_before = len(Chem.GetMolFrags(mol))
        try:
            split = _METALS.Disconnect(mol)
            Chem.SanitizeMol(split)       # recompute rings/valences after bonds were removed
            mol = split
        except Exception:                 # keep the molecule as written rather than drop it,
            pass                          # so both modes contain the same molecules
        return mol, len(Chem.GetMolFrags(mol)) != n_before
    if len(Chem.GetMolFrags(mol)) > 1:
        return _LARGEST.choose(mol), True
    return mol, False


def brics_partition(mol, allowed_atoms=None):
    """Split a molecule at BRICS bonds, keeping atom indices.

    Unlike BRICS.BRICSDecompose (which returns a set of unique SMILES), this keeps
    duplicate fragments, maps every atom to its fragment, and records which bonds
    were broken, which is what you need to build fragment-level edges.

    Returns
        groups:       list of tuples of atom indices, one per fragment
        frag_of_atom: list, fragment index for each atom
        broken:       list of (atom_i, atom_j) for each broken bond
    """
    pairs = sorted({tuple(sorted(p)) for p, _ in BRICS.FindBRICSBonds(mol)})
    if allowed_atoms is not None:           # ionic mode: counter-ions stay whole
        pairs = [p for p in pairs if p[0] in allowed_atoms and p[1] in allowed_atoms]
    bond_ids = [mol.GetBondBetweenAtoms(i, j).GetIdx() for i, j in pairs]
    pieces = Chem.FragmentOnBonds(mol, bond_ids, addDummies=False) if bond_ids else mol
    groups = list(Chem.GetMolFrags(pieces, asMols=False, sanitizeFrags=False))
    frag_of_atom = [None] * mol.GetNumAtoms()
    for f, group in enumerate(groups):
        for a in group:
            frag_of_atom[a] = f
    return groups, frag_of_atom, pairs


LABEL_SCHEME = "extended"   # "extended": functional_groups.label_fragment (curated + rings + Ertl)
                            # "basic": the original short SMARTS list below


def label_version():
    if LABEL_SCHEME == "extended":
        from functional_groups import LABEL_VERSION
        return LABEL_VERSION
    return "v1"


def functional_groups(mol, group):
    """Readable labels for one fragment, matched in the parent molecule."""
    if LABEL_SCHEME == "extended":
        from functional_groups import label_fragment
        return label_fragment(mol, group) or ["other"]
    atoms = set(group)
    labels = [name for name, patt in _FG_PATTERNS.items()
              if any(set(m) <= atoms for m in mol.GetSubstructMatches(patt))]
    drop = set().union(*(SUPPRESS.get(l, set()) for l in labels))
    labels = [l for l in labels if l not in drop]
    if labels:
        return labels
    if len(group) == 1:   # single-atom linker left between two BRICS cuts
        sym = mol.GetAtomWithIdx(group[0]).GetSymbol()
        return [f"{ELEMENT_NAMES.get(sym, sym)} linker atom"]
    if any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in group):
        return ["aromatic ring"]
    if any(mol.GetAtomWithIdx(a).IsInRing() for a in group):
        return ["aliphatic ring"]
    if all(mol.GetAtomWithIdx(a).GetSymbol() == "C" for a in group):
        return ["alkyl chain"]
    return ["other"]


def fragment_features(mol, group, logp_contribs, mr_contribs, tpsa_contribs):
    """Morgan fingerprint of the fragment (centered on its atoms) + summed atom-level
    descriptor contributions computed on the parent molecule."""
    fp = _MORGAN.GetFingerprintAsNumPy(mol, fromAtoms=list(group)).astype(np.float32)
    atoms = [mol.GetAtomWithIdx(a) for a in group]
    desc = np.array([
        len(group),                                              # heavy atoms
        sum(a.GetIsAromatic() for a in atoms) / len(group),      # aromatic fraction
        sum(a.IsInRing() for a in atoms) / len(group),           # ring-atom fraction
        sum(a.GetSymbol() == "N" for a in atoms),
        sum(a.GetSymbol() == "O" for a in atoms),
        sum(a.GetSymbol() == "S" for a in atoms),
        sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in atoms),
        sum(a.GetTotalNumHs() for a in atoms),
        sum(logp_contribs[a] for a in group),                    # Crippen logP contribution
        sum(mr_contribs[a] for a in group),                      # Crippen molar refractivity
        sum(tpsa_contribs[a] for a in group),                    # polar surface area contribution
    ], dtype=np.float32)
    return np.concatenate([fp, desc])


def bond_features(bond, ionic_mode=False):
    f = [float(bond.GetBondType() == t) for t in BOND_TYPES] + [float(bond.GetIsConjugated())]
    return f + [0.0] if ionic_mode else f


IONIC_EDGE = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def main_component(mol, components):
    """The parent molecule: the largest component that is not a known counter-ion or salt
    partner (so e.g. tosylic acid never wins over the drug it is paired with)."""
    def known(c):
        return Chem.MolFragmentToSmiles(mol, atomsToUse=list(c), canonical=True) in ION_NAMES
    cands = [i for i, c in enumerate(components) if not known(c)] or list(range(len(components)))
    return max(cands, key=lambda i: (len(components[i]),
                                     sum(mol.GetAtomWithIdx(a).GetSymbol() == "C" for a in components[i]), -i))


def ion_name(mol, comp_atoms):
    smi = Chem.MolFragmentToSmiles(mol, atomsToUse=list(comp_atoms), canonical=True)
    if smi in ION_NAMES:
        return ION_NAMES[smi]
    charge = sum(mol.GetAtomWithIdx(a).GetFormalCharge() for a in comp_atoms)
    return "cation" if charge > 0 else "anion" if charge < 0 else "salt partner / co-crystal component"


def ionic_links(mol, components, main, frag_of_atom):
    """Rules for linking each non-main component (counter-ion, salt partner, solvent) to the
    rest of the molecule, since no covalent bond connects them:
      1. charged component -> fragments holding atoms of opposite formal charge
         (prefer the main component; otherwise any other component)
      2. neutral component (e.g. 'Cl' for an HCl salt, an acid partner, water) ->
         the main-component fragment holding the most basic nitrogen (amine first,
         then aromatic N); if there is none, the main component's largest fragment
    Returns a set of (fragment_u, fragment_v) pairs."""
    links = set()
    main_atoms = set(components[main])

    def frag_sizes():
        sizes = {}
        for a in main_atoms:
            sizes[frag_of_atom[a]] = sizes.get(frag_of_atom[a], 0) + 1
        return sizes

    for ci, comp in enumerate(components):
        if ci == main:
            continue
        src = frag_of_atom[comp[0]]            # non-main components are single nodes
        charge = sum(mol.GetAtomWithIdx(a).GetFormalCharge() for a in comp)
        targets = []
        if charge != 0:
            sign = 1 if charge > 0 else -1
            opposite = [a.GetIdx() for a in mol.GetAtoms()
                        if a.GetFormalCharge() * sign < 0 and a.GetIdx() not in comp]
            in_main = [a for a in opposite if a in main_atoms]
            targets = sorted({frag_of_atom[a] for a in (in_main or opposite)})
        if not targets:
            for patt in (_BASIC_N, _AROM_N):
                hits = [m[0] for m in mol.GetSubstructMatches(patt) if m[0] in main_atoms]
                if hits:
                    targets = [frag_of_atom[hits[0]]]
                    break
        if not targets:
            sizes = frag_sizes()
            targets = [max(sizes, key=sizes.get)]
        for t in targets:
            if t != src:
                links.add((src, t))
    return links


def fragment_graph(smiles, ionic=False):
    """Everything needed to build one fragment graph, as plain Python/numpy.
    Returns None for invalid SMILES. See the module docstring for the two modes."""
    mol, changed = prepare_mol(smiles, ionic)
    if mol is None:
        return None

    components = [list(c) for c in Chem.GetMolFrags(mol)]
    main = main_component(mol, components)
    allowed = set(components[main]) if ionic else None
    groups, frag_of_atom, broken = brics_partition(mol, allowed)
    comp_of_atom = {a: ci for ci, c in enumerate(components) for a in c}

    crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    logp = [c[0] for c in crippen]
    mr = [c[1] for c in crippen]
    tpsa = list(rdMolDescriptors._CalcTPSAContribs(mol))

    texts, labels, counterion = [], [], []
    for group in groups:
        frag_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=list(group), canonical=True)
        is_ci = ionic and comp_of_atom[group[0]] != main
        if is_ci:
            fg = [ion_name(mol, group)]
            texts.append(f"{frag_smi} (counter-ion: {fg[0]})")
        else:
            fg = functional_groups(mol, group)
            texts.append(f"{frag_smi} ({', '.join(fg)})")
        labels.append(fg)
        counterion.append(is_ci)

    x = np.stack([fragment_features(mol, g, logp, mr, tpsa) for g in groups])
    if ionic:   # net formal charge of the fragment, and whether it is a counter-ion node
        extra = np.array([[sum(mol.GetAtomWithIdx(a).GetFormalCharge() for a in g), float(ci)]
                          for g, ci in zip(groups, counterion)], dtype=np.float32)
        x = np.concatenate([x, extra], axis=1)

    edges, edge_attr = [], []
    for i, j in broken:
        u, v = frag_of_atom[i], frag_of_atom[j]
        feats = bond_features(mol.GetBondBetweenAtoms(i, j), ionic)
        edges += [(u, v), (v, u)]
        edge_attr += [feats, feats]
    n_ionic = 0
    if ionic and len(components) > 1:
        for u, v in sorted(ionic_links(mol, components, main, frag_of_atom)):
            edges += [(u, v), (v, u)]
            edge_attr += [IONIC_EDGE, IONIC_EDGE]
            n_ionic += 1

    return {
        "smiles": Chem.MolToSmiles(mol),
        "ionic": ionic,
        "salt_stripped": changed and not ionic,
        "metal_disconnected": changed and ionic,
        "n_components": len(components),
        "n_counterion_nodes": sum(counterion),
        "n_ionic_edges": n_ionic,
        "num_fragments": len(groups),
        "fragment_text": texts,
        "fragment_labels": labels,
        "counterion": counterion,
        "atom_groups": [list(g) for g in groups],
        "frag_of_atom": frag_of_atom,
        "x": x,
        "edges": edges,
        "edge_attr": edge_attr,
    }


# ----------------------------------------------------------------------------- data

CSV_COLUMNS = {
    # name: (SMILES column, label column) in the standard MoleculeNet CSVs
    "BBBP": ("smiles", "p_np"),
    "BACE": ("mol", "Class"),
    "ESOL": ("smiles", "measured log solubility in mols per litre"),
}


def scaffold_split(smiles_list, frac_train=0.8, frac_valid=0.1):
    """Deterministic Bemis-Murcko scaffold split (same procedure as DeepChem's
    ScaffoldSplitter): group by scaffold, largest groups go to train first."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    groups = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        groups.setdefault(scaf, []).append(i)
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = sum(len(g) for g in ordered)
    train, valid, test = [], [], []
    for g in ordered:
        if len(train) + len(g) <= frac_train * n:
            train += g
        elif len(valid) + len(g) <= frac_valid * n:
            valid += g
        else:
            test += g
    return train, valid, test


def load_split_csv(name, csv_path):
    """Load a MoleculeNet CSV you already have and scaffold-split it locally."""
    smi_col, y_col = CSV_COLUMNS[name]
    df = pd.read_csv(csv_path)
    smiles, ys = df[smi_col].astype(str).tolist(), df[y_col].tolist()
    idx = scaffold_split(smiles)
    splits = {s: [(smiles[i], ys[i]) for i in ids]
              for s, ids in zip(("train", "valid", "test"), idx)}
    return splits, DATASETS[name][1]


CSV_PATHS = {}   # filled from --csv arguments


def load_split(name):
    """MoleculeNet via DeepChem with a scaffold split, or a local CSV if given."""
    if name in CSV_PATHS:
        return load_split_csv(name, CSV_PATHS[name])
    import deepchem as dc
    loader_name, task_type = DATASETS[name]
    tasks, (train, valid, test), _ = getattr(dc.molnet, loader_name)(
        featurizer="Raw", splitter="scaffold", transformers=[])
    splits = {}
    for split_name, ds in (("train", train), ("valid", valid), ("test", test)):
        y = ds.y[:, 0] if ds.y.ndim > 1 else ds.y      # BBBP, BACE, ESOL are single-task
        splits[split_name] = list(zip(ds.ids, y))
    return splits, task_type


def to_pyg(g, y, task_type):
    import torch
    from torch_geometric.data import Data
    edge_index = (torch.tensor(g["edges"], dtype=torch.long).t().contiguous()
                  if g["edges"] else torch.zeros((2, 0), dtype=torch.long))
    edge_dim = len(BOND_TYPES) + 1 + int(g["ionic"])
    edge_attr = (torch.tensor(g["edge_attr"], dtype=torch.float)
                 if g["edge_attr"] else torch.zeros((0, edge_dim)))
    return Data(
        x=torch.tensor(g["x"], dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(y)], dtype=torch.float),
        smiles=g["smiles"],
        fragment_text=g["fragment_text"],
        atom_groups=g["atom_groups"],
        counterion=torch.tensor(g["counterion"], dtype=torch.bool),
        ionic=g["ionic"],
        label_version=label_version(),
        num_nodes=g["num_fragments"],
    )


# ----------------------------------------------------------------------------- main

def mode_paths(ionic):
    """Output locations for each mode, so the two variants never overwrite each other."""
    return ("frag_graphs_ions", "fragment_stats_ions.csv") if ionic else ("frag_graphs", "fragment_stats.csv")


def run(names, build, out_dir=None, ionic=False):
    default_dir, stats_path = mode_paths(ionic)
    out_dir = out_dir or default_dir
    rows = []
    for name in names:
        splits, task_type = load_split(name)
        counts, stripped, invalid, multi, ci_nodes, metal = [], 0, 0, 0, 0, 0
        for split_name, items in splits.items():
            graphs = []
            for smiles, y in items:
                g = fragment_graph(smiles, ionic)
                if g is None:
                    invalid += 1
                    continue
                counts.append(g["num_fragments"])
                stripped += g["salt_stripped"]
                multi += g["n_components"] > 1
                ci_nodes += g["n_counterion_nodes"]
                metal += g["metal_disconnected"]
                if build:
                    graphs.append(to_pyg(g, y, task_type))
            if build:
                import torch
                os.makedirs(out_dir, exist_ok=True)
                torch.save(graphs, os.path.join(out_dir, f"{name}_{split_name}.pt"))
                with open(os.path.join(out_dir, f"{name}_LABELS.txt"), "w") as f:
                    f.write(label_version())
        c = np.array(counts)
        row = {
            "dataset": name,
            "mode": "BRICS + ionic" if ionic else "BRICS (salts stripped)",
            "task": task_type,
            "molecules": len(c),
            "train/valid/test": "/".join(str(len(v)) for v in splits.values()),
            "mean_fragments": c.mean(),
            "std_fragments": c.std(),
            "median_fragments": float(np.median(c)),
            "max_fragments": int(c.max()),
            "pct_le_12": 100 * (c <= MAX_FRAGMENTS_FOR_EXACT_SHAPLEY).mean(),
            "pct_single_fragment": 100 * (c == 1).mean(),
            "salts_stripped": stripped,
            "invalid_smiles": invalid,
        }
        if ionic:
            row.update(pct_with_counterions=100 * multi / len(c), counterion_nodes=ci_nodes,
                       metal_bonds_split=metal)
        rows.append(row)
        print(f"done: {name}")

    df = pd.DataFrame(rows)
    if os.path.exists(stats_path):    # keep rows for datasets not rebuilt this time
        old = pd.read_csv(stats_path)
        df = pd.concat([old[~old["dataset"].isin(df["dataset"])], df], ignore_index=True)
    df.to_csv(stats_path, index=False)
    pd.set_option("display.width", 200)
    print("\n", df.round(2).to_string(index=False))
    print("\nLaTeX rows (Dataset & Task & #Mol & Frags mean±std & Max & % ≤12 & % single):")
    for r in rows:
        print(f"{r['dataset']} & {r['task']} & {r['molecules']} & "
              f"{r['mean_fragments']:.1f} $\\pm$ {r['std_fragments']:.1f} & {r['max_fragments']} & "
              f"{r['pct_le_12']:.1f} & {r['pct_single_fragment']:.1f} \\\\")


def demo(smiles, ionic=False):
    g = fragment_graph(smiles, ionic)
    if g is None:
        print("Invalid SMILES")
        return
    note = ("  (counter-ion removed)" if g["salt_stripped"] else
            f"  ({g['n_components']} components, {g['n_ionic_edges']} ionic edges)" if ionic else "")
    print(f"{g['smiles']}  ->  {g['num_fragments']} fragments{note}")
    for i, t in enumerate(g["fragment_text"]):
        print(f"  node {i}: {t}")
    print("  edges:", sorted({tuple(sorted(e)) + (("ionic",) if a[-1] == 1.0 and ionic else ())
                              for e, a in zip(g["edges"], g["edge_attr"])}))
    print("  node feature size:", g["x"].shape[1], "| edge feature size:", len(g["edge_attr"][0]) if g["edge_attr"] else "-")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=str, help="print the fragment graph for one SMILES")
    ap.add_argument("--stats", action="store_true", help="fragment statistics for the datasets")
    ap.add_argument("--build", action="store_true", help="also save PyG graphs (needs torch_geometric)")
    ap.add_argument("--ionic", action="store_true",
                    help="BRICS + ionic decomposition (keep counter-ions as nodes); default strips them")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--out", default=None, help="graph folder (default frag_graphs or frag_graphs_ions)")
    ap.add_argument("--csv", nargs="*", default=[], metavar="NAME=PATH",
                    help="use local CSVs instead of DeepChem, e.g. BBBP=BBBP.csv BACE=bace.csv ESOL=delaney-processed.csv")
    args = ap.parse_args()
    for item in args.csv:
        k, v = item.split("=", 1)
        CSV_PATHS[k.upper()] = v
    if args.demo:
        demo(args.demo, args.ionic)
    elif args.stats or args.build:
        run(args.datasets, args.build, args.out, args.ionic)
    else:
        ap.print_help()
