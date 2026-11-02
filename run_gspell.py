"""GSPELL (graph-level) explanations on BBBP test molecules, compared with GNNExplainer,
a text-only LLM, and a random baseline. Needs a GPU and Hugging Face access to Llama.

    export HF_TOKEN=hf_...
    python run_gspell.py --dataset BBBP --n_eval 30

Outputs: gspell_{DATASET}_summary.csv, gspell_{DATASET}_explanations.json
"""

import argparse
import os

import torch

from frag_gin import load_graphs
from gspell_mol import (collect_embeddings, load_llm, load_spec, run_explanations, save_records,
                        text_embeddings, train_projector)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BBBP", help="BBBP, BACE or ESOL")
    ap.add_argument("--seed", type=int, default=0, help="which trained GIN checkpoint to explain")
    ap.add_argument("--llm", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--k", type=int, default=5, help="soft-prompt tokens per embedding")
    ap.add_argument("--projector_epochs", type=int, default=30)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--no_text_only", action="store_true", help="skip the text-only LLM baseline")
    ap.add_argument("--ionic", action="store_true", help="explain the BRICS + ionic model")
    args = ap.parse_args()

    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])
    device = torch.device("cuda")

    gdir, cdir = ("frag_graphs_ions", "checkpoints_ions") if args.ionic else ("frag_graphs", "checkpoints")
    tag = "_ions" if args.ionic else ""
    spec = load_spec(args.dataset, args.seed, device, graph_dir=gdir, ckpt_dir=cdir)
    gin = spec.gin
    train, _, test = load_graphs(args.dataset, gdir)
    tok, llm = load_llm(args.llm)
    llm_dim = llm.get_input_embeddings().weight.shape[1]
    token_norm = llm.get_input_embeddings().weight.float().norm(dim=1).mean().item()

    Hn, Tn, Hg, Tg = collect_embeddings(gin, train, device)
    T = text_embeddings(Tn + Tg, tok, llm)
    proj, hist = train_projector(torch.cat([Hn, Hg]), T, llm_dim, token_norm, device,
                                 k=args.k, epochs=args.projector_epochs)
    print(f"projector loss: {hist[0]:.3f} -> {hist[-1]:.3f}")
    torch.save(proj.state_dict(), f"{cdir}/{args.dataset}_projector_k{args.k}.pt")

    df, summary = run_explanations(spec, proj, tok, llm, test, args.dataset, device,
                                   n_eval=args.n_eval, text_only_baseline=not args.no_text_only,
                                   records_path=f"explain_{args.dataset}{tag}.jsonl")
    summary.to_csv(f"gspell_{args.dataset}{tag}_summary.csv", index=False)
    save_records(df, f"gspell_{args.dataset}{tag}_explanations.json")
    print(summary.round(2).to_string(index=False))
