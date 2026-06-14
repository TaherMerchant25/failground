"""
Main experiment runner for FAILGROUND.

Usage:
  python run_experiment.py --benchmark alfworld --ablation none
  python run_experiment.py --benchmark intercode --ablation A4
  python run_experiment.py --benchmark all --episodes 100
"""

from __future__ import annotations
import argparse
import json
import os
import yaml
from pathlib import Path

from sentence_transformers import SentenceTransformer
from openai import OpenAI

from fakg.fakg import FAKG
from fakg.ccsr import CCSR
from fakg.afm import AFM
from fakg.mast import MASTClassifier
from pipeline.failure_miner import FailureMiner
from agent.react_agent import FAILGROUNDReActAgent
from eval.metrics import EpisodeResult, compute_metrics
from tools.io_file import create_if_not_exist


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_system(cfg: dict, ablation: str):
    fakg = FAKG()

    mast = MASTClassifier(
        model=cfg["mast_classifier"]["model"],
        base_url=cfg["mast_classifier"]["base_url"],
        api_key=cfg["mast_classifier"].get("api_key", "EMPTY"),
    )

    embedder = SentenceTransformer(cfg["embedding"]["model"])
    embed_fn = lambda text: embedder.encode(text).tolist()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", cfg["agent"]["api_key"]),
        base_url=cfg["agent"]["base_url"],
    )
    llm_fn = lambda prompt: client.chat.completions.create(
        model=cfg["agent"]["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    ).choices[0].message.content

    ccsr = CCSR(fakg=fakg,
                fcs_threshold=cfg["ccsr"]["fcs_threshold"],
                sim_threshold=cfg["ccsr"]["sim_threshold"])

    afm = AFM(fakg=fakg,
              budget=cfg["fakg"]["budget"],
              alpha=cfg["fakg"]["afm_alpha"],
              beta=cfg["fakg"]["afm_beta"],
              gamma=0.0 if ablation == "A5" else cfg["fakg"]["afm_gamma"],
              decay_lambda=cfg["fakg"]["afm_lambda"],
              fifo_mode=(ablation == "A4"))

    miner = FailureMiner(fakg=fakg, mast=mast, llm_fn=llm_fn, embed_fn=embed_fn,
                         conceptnet_dump=cfg["sparql"].get("conceptnet_dump"))

    agent = FAILGROUNDReActAgent(fakg=fakg, ccsr=ccsr, afm=afm,
                                  llm_fn=llm_fn, embed_fn=embed_fn,
                                  max_steps=cfg["agent"]["max_steps"])
    return agent, miner, afm, fakg, embed_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark",
                        choices=["alfworld", "intercode", "webarena", "all"],
                        default="alfworld")
    parser.add_argument("--ablation", default="none",
                        choices=["none","A1","A2","A3","A4","A5","A6"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    create_if_not_exist(args.output)
    agent, miner, afm, fakg, embed_fn = build_system(cfg, args.ablation)

    print(f"Running {args.benchmark} | ablation={args.ablation} | episodes={args.episodes}")
    print("ALFWorld/InterCode/WebArena env wrappers: implement in agent/envs/")

    out_path = Path(args.output) / f"{args.benchmark}_{args.ablation}.json"
    fakg.save(str(Path(args.output) / f"fakg_{args.benchmark}_{args.ablation}.json"))
    print(f"FAKG saved. Add env wrappers to run full eval.")


if __name__ == "__main__":
    main()
