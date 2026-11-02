"""Functional-group and ring labels for fragments, plus list-free detection.

Three sources, combined in label_fragment():

1. Curated SMARTS (rule list). RDKit's FragmentDescriptors.csv (85 groups; the basis of
   RDKit's fr_* descriptors), minus entries that are metabolic-site heuristics or counts rather
   than functional groups, plus groups that list lacks (carbamate, sulfonic acid, sulfoxide,
   acyl halide, anhydride, alkene, alkyne, CF3, ...).
2. Ring names. Each ring of the fragment is matched to a table of common aromatic and
   aliphatic rings (pyridine, piperazine, ...); unknown rings are described by size/aromaticity.
3. Ertl's algorithm (J. Cheminform. 2017, 9:36) - no predefined list. Atoms are marked by
   general rules (heteroatoms; carbons in C=C, C#C or double/triple bonds to heteroatoms;
   acetal-type sp3 carbons; atoms of 3-membered heterocycles); connected marked atoms form one
   functional group, reported as a generalized SMILES type. Any group the curated list misses
   still gets a label this way ("group N=C(N)S" etc.).

Redundant labels are removed by subsumption: a label is dropped when every one of its matches
lies inside a strictly larger match of another label (so "carboxylic acid" absorbs "hydroxyl"
and "carbonyl" without hand-written exceptions).

mine_groups(smiles) counts Ertl types across a dataset: a data-driven vocabulary of the groups
that actually occur, with how often the curated list already names them.
"""

import csv
import os
from collections import Counter

from rdkit import Chem, RDConfig, RDLogger

RDLogger.DisableLog("rdApp.*")

LABEL_VERSION = "v2"   # bump when labels change; graphs/projectors/explanations are tagged with it

# ----------------------------------------------------------------------------- curated SMARTS

# RDKit entries that are not functional groups (metabolism sites, bicyclic flag, duplicates,
# overly broad counts) are skipped.
_SKIP = {"fr_C=O_noCOO", "fr_COO2", "fr_NH0", "fr_NH1", "fr_NH2", "fr_Ar_N", "fr_Ar_NH", "fr_Nhpyrrole",
         "fr_bicyclic", "fr_para_hydroxylation", "fr_allylic_oxid", "fr_aryl_methyl",
         "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_HOCCN", "fr_ketone_Topliss",
         "fr_phenol_noOrthoHbond", "fr_Al_OH_noTert", "fr_nitro_arom_nonortho", "fr_ArN",
         "fr_unbrch_alkane", "fr_benzodiazepine", "fr_COO", "fr_C=O", "fr_halogen",
         "fr_imidazole", "fr_furan", "fr_thiophene", "fr_thiazole", "fr_oxazole", "fr_pyridine",
         "fr_piperdine", "fr_piperzine", "fr_morpholine", "fr_tetrazole", "fr_benzene"}   # rings: see RING_NAMES
# Shorter display names for RDKit entries
_RENAME = {"fr_Al_OH": "aliphatic hydroxyl", "fr_Ar_OH": "aromatic hydroxyl", "fr_Al_COO": "aliphatic carboxylic acid",
           "fr_Ar_COO": "aromatic carboxylic acid", "fr_quatN": "quaternary nitrogen", "fr_Imine": "imine",
           "fr_hdrzine": "hydrazine", "fr_hdrzone": "hydrazone", "fr_N-O": "hydroxylamine",
           "fr_priamide": "primary amide", "fr_guanido": "guanidine", "fr_isocyan": "isocyanate",
           "fr_isothiocyan": "isothiocyanate", "fr_sulfide": "thioether", "fr_SH": "thiol",
           "fr_C=S": "thiocarbonyl", "fr_sulfonamd": "sulfonamide", "fr_prisulfonamd": "primary sulfonamide",
           "fr_barbitur": "barbiturate", "fr_term_acetylene": "terminal alkyne", "fr_lactam": "beta-lactam",
           "fr_lactone": "lactone", "fr_epoxide": "epoxide", "fr_phos_acid": "phosphoric acid",
           "fr_phos_ester": "phosphate ester", "fr_nitro_arom": "nitroarene", "fr_dihydropyridine": "dihydropyridine",
           "fr_alkyl_carbamate": "alkyl carbamate", "fr_alkyl_halide": "alkyl halide", "fr_methoxy": "methoxy",
           "fr_aniline": "aniline (aryl amine)", "fr_ether": "ether", "fr_nitroso": "nitroso",
           "fr_amidine": "amidine", "fr_azide": "azide", "fr_azo": "azo", "fr_diazo": "diazo", "fr_imide": "imide",
           "fr_oxime": "oxime", "fr_sulfone": "sulfone", "fr_urea": "urea", "fr_nitro": "nitro"}
# Groups the RDKit list does not have
EXTRA_SMARTS = {
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "carboxylate": "[CX3](=O)[O-]",
    "carbamate": "[NX3][CX3](=O)[OX2][#6]",
    "carbonate": "[OX2][CX3](=O)[OX2]",
    "acyl halide": "[CX3](=O)[F,Cl,Br,I]",
    "anhydride": "[CX3](=O)[OX2][CX3]=O",
    "sulfonic acid": "[SX4](=O)(=O)[OX2H1,OX1-]",
    "sulfonate ester": "[SX4](=O)(=O)[OX2][#6]",
    "sulfoxide": "[$([SX3](=O)([#6])[#6]),$([SX3+]([O-])([#6])[#6])]",
    "sulfonyl": "[SX4](=O)(=O)",
    "disulfide": "[SX2][SX2]",
    "primary amine": "[NX3;H2;!$(N-C=[O,S,N]);!$(N-a);!$(N-S)][CX4]",
    "secondary amine": "[NX3;H1;!$(N-C=[O,S,N]);!$(N-a);!$(N-S)]([CX4])[CX4]",
    "tertiary amine": "[NX3;H0;!$(N-C=[O,S,N]);!$(N-a);!$(N-S);!$(N=*);!$(N#*)]([CX4])([CX4])[CX4]",
    "protonated amine (ammonium)": "[NX4+;!$(N~O)]",
    "enamine": "[NX3][CX3]=[CX3]",
    "enol ether": "[OX2]([#6])[CX3]=[CX3]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "trifluoromethyl": "C(F)(F)F",
    "aryl halide": "c[F,Cl,Br,I]",
    "acetal / ketal": "[CX4]([OX2][#6])[OX2][#6]",
    "hemiacetal": "[CX4]([OX2H])[OX2][#6]",
    "boronic acid": "[BX3]([OX2H])[OX2H]",
    "N-oxide": "[#7+;!$([#7+](=O)[O-]);!$([#7+]=O)][OX1-]",
    "ring carbonyl (aromatic C=O)": "[#6;a]=[OX1]",
}


def _load_rdkit_groups():
    path = os.path.join(RDConfig.RDDataDir, "FragmentDescriptors.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for row in csv.reader((l for l in f if l.strip() and not l.startswith("#")), delimiter="\t"):
            key, desc, smarts = row[0], row[1], row[2]
            if key in _SKIP:
                continue
            name = _RENAME.get(key) or desc.replace("Number of ", "").rstrip("s").strip().lower()
            out[name] = smarts
    return out


CURATED = {**_load_rdkit_groups(), **EXTRA_SMARTS}
_PATTERNS = {}
for _name, _s in CURATED.items():
    _p = Chem.MolFromSmarts(_s)
    if _p is not None:
        _PATTERNS[_name] = _p

# ----------------------------------------------------------------------------- rings

RING_NAMES = {
    # aromatic 6
    "c1ccccc1": "benzene ring", "c1ccncc1": "pyridine ring", "c1cncnc1": "pyrimidine ring",
    "c1cnccn1": "pyrazine ring", "c1ccnnc1": "pyridazine ring", "c1ncncn1": "triazine ring",
    "O=c1cccc[nH]1": "pyridone ring",
    # aromatic 5
    "c1cc[nH]c1": "pyrrole ring", "c1ccoc1": "furan ring", "c1ccsc1": "thiophene ring",
    "c1c[nH]cn1": "imidazole ring", "c1cn[nH]c1": "pyrazole ring", "c1cocn1": "oxazole ring",
    "c1cnoc1": "isoxazole ring", "c1cscn1": "thiazole ring", "c1cnsc1": "isothiazole ring",
    "c1nc[nH]n1": "1,2,4-triazole ring", "c1cn[nH]n1": "1,2,3-triazole ring", "c1nn[nH]n1": "tetrazole ring",
    "c1nnco1": "1,3,4-oxadiazole ring", "c1ncon1": "1,2,4-oxadiazole ring", "c1nncs1": "thiadiazole ring",
    # aliphatic
    "C1CC1": "cyclopropane", "C1CCC1": "cyclobutane", "C1CCCC1": "cyclopentane", "C1CCCCC1": "cyclohexane",
    "C1CCCCCC1": "cycloheptane", "C1CCNC1": "pyrrolidine ring", "C1CCNCC1": "piperidine ring",
    "C1CNCCN1": "piperazine ring", "C1COCCN1": "morpholine ring", "C1CCOC1": "tetrahydrofuran ring",
    "C1CCOCC1": "tetrahydropyran ring", "C1CSCN1": "thiazolidine ring", "C1CO1": "oxirane ring",
    "C1CN1": "aziridine ring", "C1COCO1": "dioxolane ring", "C1CCSC1": "tetrahydrothiophene ring",
    "C1=CCCCC1": "cyclohexene", "C1CNC1": "azetidine ring", "C1=CCCC1": "cyclopentene",
    "C1=CNC=CC1": "1,4-dihydropyridine ring", "C1=CCNCC1": "tetrahydropyridine ring",
    "O=C1CCCN1": "pyrrolidinone (lactam) ring", "C1CCNCC1": "piperidine ring",
}


def _canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else smi


RING_NAMES = {_canon(k): v for k, v in RING_NAMES.items()}   # keys in RDKit canonical form


def _ring_key(mol, ring):
    """Ring atoms only, all exocyclic substituents removed, charges and H counts reset."""
    em = Chem.RWMol()
    idx = {}
    for a in ring:
        at = mol.GetAtomWithIdx(a)
        n = Chem.Atom(at.GetAtomicNum())
        n.SetIsAromatic(at.GetIsAromatic())
        idx[a] = em.AddAtom(n)
    for i, a in enumerate(ring):
        b = ring[(i + 1) % len(ring)]
        bond = mol.GetBondBetweenAtoms(a, b)
        em.AddBond(idx[a], idx[b], bond.GetBondType())
    m = em.GetMol()
    try:
        m.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
        smi = Chem.MolToSmiles(m)
        # aromatic N with implicit H: try the [nH] form so pyrrole/imidazole-type rings match the table
        if smi not in RING_NAMES:
            for alt in (smi.replace("n", "[nH]", 1),):
                mm = Chem.MolFromSmiles(alt)
                if mm is not None and Chem.MolToSmiles(mm) in RING_NAMES:
                    return Chem.MolToSmiles(mm)
        return smi
    except Exception:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=list(ring))


def ring_labels(mol, atoms):
    atoms = set(atoms)
    ri = mol.GetRingInfo()
    rings = [r for r in ri.AtomRings() if set(r) <= atoms]
    labels = []
    for r in rings:
        key = _ring_key(mol, r)
        if key in RING_NAMES:
            labels.append(RING_NAMES[key])
        else:
            arom = all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)
            het = sorted({mol.GetAtomWithIdx(a).GetSymbol() for a in r} - {"C"})
            labels.append(f"{len(r)}-membered {'aromatic' if arom else 'aliphatic'} "
                          + (f"{'/'.join(het)}-heterocycle" if het else "carbocycle"))
    fused = any(len(set(r1) & set(r2)) >= 2 for i, r1 in enumerate(rings) for r2 in rings[i + 1:])
    if fused:
        labels.append("fused ring system")
    return labels


# ----------------------------------------------------------------------------- Ertl IFG

def ertl_groups(mol):
    """Ertl's functional-group identification. Returns a list of (atom set, type SMILES)."""
    marked = set()
    for at in mol.GetAtoms():
        if at.GetAtomicNum() not in (1, 6):                       # heteroatoms
            marked.add(at.GetIdx())
    for b in mol.GetBonds():                                      # C=C, C#C, C=X, C#X (non-aromatic)
        if b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) and not b.GetIsAromatic():
            marked.update((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    for at in mol.GetAtoms():                                     # acetal-type sp3 carbons
        if at.GetAtomicNum() == 6 and at.GetHybridization() == Chem.HybridizationType.SP3:
            het = [n for n in at.GetNeighbors() if n.GetAtomicNum() in (7, 8, 16)
                   and all(bb.GetBondType() == Chem.BondType.SINGLE for bb in n.GetBonds())]
            if len(het) >= 2:
                marked.add(at.GetIdx())
    for r in mol.GetRingInfo().AtomRings():                       # oxirane / aziridine / thiirane
        if len(r) == 3 and any(mol.GetAtomWithIdx(a).GetAtomicNum() in (7, 8, 16) for a in r):
            marked.update(r)
    # connected components of marked atoms
    groups, seen = [], set()
    for a in marked:
        if a in seen:
            continue
        stack, comp = [a], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack += [n.GetIdx() for n in mol.GetAtomWithIdx(x).GetNeighbors() if n.GetIdx() in marked]
        seen |= comp
        groups.append(comp)
    out = []
    for comp in groups:
        try:
            t = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(comp), canonical=True)
        except Exception:
            t = "?"
        out.append((comp, t))
    return out


# ----------------------------------------------------------------------------- labelling

def _subsume(matches):
    """matches: {label: [atom sets]}. Drop a label when every one of its matches lies inside a
    larger match of another label, or coincides with a match of a more specific label
    (longer SMARTS), e.g. 'sulfonamide' gives way to 'primary sulfonamide'."""
    spec = {l: len(CURATED.get(l, "")) for l in matches}
    keep = []
    for lab, ms in matches.items():
        if "$(" in CURATED.get(lab, ""):      # recursive SMARTS match only an anchor atom: keep
            keep.append(lab)
            continue
        def absorbed(m):
            for l2, ms2 in matches.items():
                if l2 == lab:
                    continue
                for o in ms2:
                    if m < o or (m == o and spec[l2] > spec[lab]):
                        return True
            return False
        if all(absorbed(m) for m in ms):
            continue
        keep.append(lab)
    return keep


ELEMENT_NAMES = {"N": "nitrogen", "O": "oxygen", "S": "sulfur", "C": "carbon", "P": "phosphorus"}


def label_fragment(mol, atoms, max_labels=6):
    """Readable labels for the fragment made of `atoms` (indices into mol):
      - single-atom fragments: '<element> linker atom'
      - rings in the fragment (named when known)
      - curated groups fully inside the fragment
      - 'part of <group>' for curated groups that BRICS cut through (e.g. an amide split
        between its C=O and N fragments)
      - Ertl groups (list-free) for anything the curated list did not cover"""
    atoms = set(atoms)
    inside, crossing = {}, {}
    for name, patt in _PATTERNS.items():
        for m in mol.GetSubstructMatches(patt):
            m = set(m)
            if m <= atoms:
                inside.setdefault(name, []).append(m)
            elif m & atoms:
                crossing.setdefault(name, []).append(m)
    fg_labels = _subsume(inside)
    marked = set().union(*(c for c, _ in ertl_groups(mol))) if mol.GetNumAtoms() else set()
    inside_sets = [m for l in fg_labels for m in inside[l]]
    # 'part of X' only when the fragment holds part of X's functional core (a heteroatom or a
    # multiply bonded carbon), not just a carbon it is attached to, and nothing inside already covers it
    crossing = {l: [m for m in ms if (m & atoms & marked) and not any((m & atoms) <= im for im in inside_sets)]
                for l, ms in crossing.items()}
    crossing = {l: ms for l, ms in crossing.items() if ms}
    part_labels = [f"part of {l}" for l in _subsume(crossing) if l not in fg_labels]
    named_atoms = set()
    for l in fg_labels:
        for m in inside[l]:
            named_atoms |= m
    for l in crossing:
        for m in crossing[l]:
            named_atoms |= (m & atoms)
    ring_l = ring_labels(mol, atoms)
    ring_atoms = {a for r in mol.GetRingInfo().AtomRings() if set(r) <= atoms for a in r}
    extra = []
    for comp, t in ertl_groups(mol):
        local = comp & atoms
        if not local or local & named_atoms:
            continue
        if local <= ring_atoms and all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in local):
            continue      # aromatic ring heteroatoms: already described by the ring name
        if len(local) == 1 and mol.GetAtomWithIdx(next(iter(local))).GetSymbol() in ("F", "Cl", "Br", "I"):
            extra.append("halogen")
            continue
        if comp <= atoms:
            extra.append(f"group {t}")
        else:
            part = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(local), canonical=True)
            extra.append(f"part of group {t}" if len(t) <= 12 else f"group {part} (cut)")
    head = []
    if len(atoms) == 1:
        sym = mol.GetAtomWithIdx(next(iter(atoms))).GetSymbol()
        head = [f"{ELEMENT_NAMES.get(sym, sym)} linker atom"]
    fg_labels = [l for l in fg_labels if not any(l in r for r in ring_l)]   # e.g. 'dihydropyridine'
    labels = list(dict.fromkeys(head + ring_l + fg_labels + part_labels + extra))
    if not labels and all(mol.GetAtomWithIdx(a).GetSymbol() == "C" for a in atoms):
        labels = ["alkyl chain"]
    return labels[:max_labels]


def mine_groups(smiles_list, top=40):
    """Data-driven vocabulary: Ertl group types found in a set of molecules, how many molecules
    contain each, and whether the curated list names them. Returns a pandas DataFrame."""
    import pandas as pd
    counts, named = Counter(), Counter()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        seen = set()
        cover = set(a for p in _PATTERNS.values() for m in mol.GetSubstructMatches(p) for a in m)
        named_rings = set(a for r in mol.GetRingInfo().AtomRings() if _ring_key(mol, r) in RING_NAMES for a in r)
        for comp, t in ertl_groups(mol):
            if t in seen:
                continue
            seen.add(t)
            counts[t] += 1
            # same rule as label_fragment: overlapping a curated match, or sitting in a named ring
            if comp & cover or comp <= named_rings:
                named[t] += 1
    rows = [{"ertl_type": t, "molecules": c, "pct_named_by_curated": 100 * named[t] / c}
            for t, c in counts.most_common(top)]
    return pd.DataFrame(rows)
