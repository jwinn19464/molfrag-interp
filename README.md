# Fragments to Narratives: GSPELL for Molecular Property Prediction

In this project I took **GSPELL** ([Baghershahi et al., 2026](https://arxiv.org/abs/2508.07117)), which is an LLM-based explainer for graph neural networks, and tried to adapt it for molecules. The original GSPELL works on node classification for text-attributed graphs, but here it is used for **graph-level** explanation instead. To do this, each molecule gets broken up into BRICS fragments, and then those fragments are used as the nodes of a graph. Each node has text attached to it (the fragment SMILES and its functional groups), so basically every molecule ends up being a small text-attributed graph. Then a GIN is trained on these fragment graphs, and a frozen Llama-3.1-8B-Instruct explains the GIN, using the GIN's embeddings as soft prompts.
Later on, other interpretability methods, such as SHAP and GradCAM were examined. Future work could be to extend this to hypergraph neural network approaches. The motivation behind this project is to examine the reasoning behind predictions given by GNN, ML, deep learning, and other AI models used to predict molecular properties such as toxicity, blood-brain barrier penetration, and more. It is crucial to know the reasoning behind these predictions in order to ensure that models are trustworthy and reflective of reality.

![Pipeline](figures/pipeline.png)

## Setup

You need Python 3.10+ and an NVIDIA GPU (CUDA). I used Google Colab with an L4 GPU for this.

```bash
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate frag-gspell
```

The explanations use `meta-llama/Llama-3.1-8B-Instruct`, which is gated on Hugging Face, so you have to request access on the model page first (it usually gets approved within a day or so). After that, set your token:

```bash
export HF_TOKEN=hf_...
```

## Reproduce

```bash
python prepare_data.py   # download BBBP/BACE/ESOL, scaffold split 80/10/10, build fragment graphs, write fragment_stats.csv
python train_gin.py      # fragment-graph GIN, 3 seeds per dataset -> gin_results.csv, checkpoints/
python run_gspell.py --dataset BBBP --n_eval 30   # explanations + fidelity/size vs. baselines -> gspell_BBBP_summary.csv
                                                  # also BACE, or ESOL (regression: classed at the training median)
```

If you already have the CSVs, you can run `prepare_data.py --data_dir <folder>` and it will use those instead of downloading them again.

**Ionic mode.** If you add `--ionic` to `prepare_data.py`, `train_gin.py` and `run_gspell.py`, the counter-ions will be kept as graph nodes instead of getting removed. The outputs for this go into `frag_graphs_ions/` and `checkpoints_ions/`, so that way both modes can be compared with each other.

All of the same steps are also in the Colab notebooks in `notebooks/`. **`run_all.ipynb` runs everything in one session** (data, models, diagnostics, LLM explanations, tables, figures, and export). It skips anything that was already done, so if Colab disconnects you can just rerun it. The other notebooks below are the same stages but split up:

| Notebook | What it does |
|---|---|
| `run_all.ipynb` | runs all of the stages for both modes in one session; can be resumed; writes `results/latex_rows.txt` and a zip export |
| `prelim_report_pipeline.ipynb` | fragment graphs, statistics table, fragment histogram, GIN training |
| `gspell_bbbp.ipynb` | projector training, GSPELL explanations on BBBP, and comparison with GNNExplainer |
| `ionic_mode.ipynb` | builds the BRICS + ionic graphs and compares the statistics and GIN results with the default mode |
| `diagnostics.ipynb` | for both modes: confusion matrices (classification), parity plots (regression), t-SNE/UMAP of the molecule and fragment embeddings, and readout-attention maps; does not need the LLM |
| `explain_more.ipynb` | BBBP, BACE and ESOL with more molecules; atom-GIN baseline; Grad-CAM (fragment and atom level) and readout attention; agreement analysis; RDKit drawings |

## Files

| File | Purpose |
|---|---|
| `fragment_graphs.py` | BRICS partition with the atom mapping, fragment text and features, fragment edges, and scaffold split |
| `functional_groups.py` | labels for the fragments: RDKit's functional-group list plus some additions, ring names, Ertl's method for finding groups without a list, and mining groups across the dataset |
| `faithfulness.py` | exact fragment Shapley values, removal counterfactuals, deletion curves, how much each explainer agrees with Shapley, and the QA faithfulness test (with vs. without soft prompts) |
| `frag_gin.py` | the fragment-graph GIN (GINE layers, attention readout), plus training and evaluation |
| `gspell_mol.py` | projector, hybrid soft prompts, LLM explanations, fidelity, and the GNNExplainer and random baselines |
| `viz_mol.py` | confusion matrix, parity plot, t-SNE/UMAP, attention summaries and maps, explanation heatmaps, and the soft-prompt alignment plot |
| `cam_mol.py` | Grad-CAM on the fragment and atom GINs, readout attention, the atom-level GIN, agreement metrics, and RDKit drawings |
| `prepare_data.py`, `train_gin.py`, `run_gspell.py` | the command-line scripts |

## Method summary

- **Fragment graph.** The BRICS bonds get cut (with `FindBRICSBonds` + `FragmentOnBonds`), and duplicate fragments are kept, along with the map of which atom goes to which fragment. The nodes are the fragments and the edges are the bonds that were cut (bond type + conjugation). For the node features, there is a 512-bit Morgan fingerprint centered on the fragment's atoms, plus some atom-level descriptors added up (Crippen logP/MR, TPSA, heteroatom counts, aromatic fraction, etc.). Counter-ions are removed, and if a molecule doesn't have any BRICS bonds it just becomes a graph with one node.
- **Ionic mode (optional).** In this mode every component of the SMILES is kept. Metal-heteroatom bonds that are written as covalent get split into ions (using RDKit's `MetalDisconnector`). BRICS cuts are only done on the parent molecule, which is the largest component that isn't a known counter-ion, and each counter-ion or salt partner becomes one named node. Each counter-ion is connected with an ionic edge to the fragments that have the opposite formal charge. For neutral partners, like `.Cl` in an HCl salt, it gets connected to the fragment that has the most basic nitrogen instead. There are also some extra features: net formal charge and a counter-ion flag on the nodes, and an ionic flag on the edges.
- **GIN.** An MLP compresses the fragment features, then there are a few GINE layers with residual connections and an attention readout. It is trained with a few different seeds and early stopping on the validation set.
- **GSPELL, graph-level.** A projector Π turns each fragment embedding and the molecule embedding into a small number of soft tokens, and it is trained with GSPELL's context-alignment and contrastive losses. The prompt has the molecule's soft prompt, the GIN's prediction, and then the text for every fragment followed by that fragment's soft prompt. The LLM answers YES or NO for whether each fragment supports the prediction, and the YES fragments make up the explanation subgraph.
- **Faithfulness (the main contribution of the project).** I computed exact fragment-level Shapley values by going through every subset of fragments (this only works for molecules that don't have too many fragments), on the GIN's output pointed toward the predicted class, and checked them with the efficiency axiom. From the same table I also got single-fragment removal counterfactuals. There are also deletion curves for each explainer's ranking, and a QA test where the answers come from the model and the LLM is asked with and without the soft prompts.
- **Evaluation.** Fidelity means the GIN still gives the same class on the explanation subgraph (for ESOL, it means staying on the same side of the training median). Explanation size is also measured. These are compared against GNNExplainer (threshold and size-matched), a text-only LLM without the soft prompts, Grad-CAM on the fragment GIN, readout attention, Grad-CAM on an atom-level GIN summed over the fragments, and a random baseline that picks half the fragments. The CAM-type explainers keep the same number of fragments as GSPELL so that the comparison is fair.

## Acknowledgements

The GSPELL design is based on the original paper and [code](https://github.com/pbaghershahi/GSPELL). The datasets are from MoleculeNet (Wu et al., 2018).
