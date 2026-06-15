"""
Main experiment runner for FAILGROUND.

Usage:
  python run_experiment.py --benchmark alfworld --ablation none --episodes 50
  python run_experiment.py --benchmark intercode --ablation A4 --episodes 50
  python run_experiment.py --benchmark all --episodes 100
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import uuid
import yaml
from pathlib import Path

from sentence_transformers import SentenceTransformer
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

from fakg.fakg import FAKG
from fakg.ccsr import CCSR
from fakg.afm import AFM
from fakg.mast import MASTClassifier
from pipeline.failure_miner import FailureMiner
from agent.react_agent import FAILGROUNDReActAgent
from agent.envs import ALFWorldEnv, InterCodeSQLEnv, WebArenaLiteEnv
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

    api_key = os.environ.get("OPENAI_API_KEY", cfg["agent"].get("api_key", ""))
    client = OpenAI(api_key=api_key, base_url=cfg["agent"]["base_url"])
    model = cfg["agent"]["model"]

    def llm_fn(prompt: str) -> str:
        try:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            ).choices[0].message.content
        except Exception as e:
            return f"[LLM error: {e}]"

    ccsr = CCSR(
        fakg=fakg,
        fcs_threshold=cfg["ccsr"]["fcs_threshold"],
        sim_threshold=cfg["ccsr"]["sim_threshold"],
    )

    afm = AFM(
        fakg=fakg,
        budget=cfg["fakg"]["budget"],
        alpha=cfg["fakg"]["afm_alpha"],
        beta=cfg["fakg"]["afm_beta"],
        gamma=0.0 if ablation == "A5" else cfg["fakg"]["afm_gamma"],
        decay_lambda=cfg["fakg"]["afm_lambda"],
        fifo_mode=(ablation == "A4"),
    )

    miner = FailureMiner(
        fakg=fakg,
        mast=mast,
        llm_fn=llm_fn,
        embed_fn=embed_fn,
        conceptnet_dump=cfg["sparql"].get("conceptnet_dump"),
    )

    agent = FAILGROUNDReActAgent(
        fakg=fakg,
        ccsr=ccsr,
        afm=afm,
        llm_fn=llm_fn,
        embed_fn=embed_fn,
        max_steps=cfg["agent"]["max_steps"],
    )
    return agent, miner, afm, fakg, embed_fn


def run_benchmark(bench: str, agent, miner, afm, fakg, embed_fn, cfg, n_episodes: int) -> dict:
    if bench == "alfworld":
        env_cls = ALFWorldEnv
        env_kwargs = {"split": "eval_out_of_distribution"}
        env_name = "alfworld"
    elif bench == "intercode":
        env_cls = InterCodeSQLEnv
        env_kwargs = {}
        env_name = "intercode"
    elif bench == "webarena":
        env_cls = WebArenaLiteEnv
        env_kwargs = {"base_url": cfg["benchmarks"]["webarena_lite"]["base_url"]}
        env_name = "webarena"
    else:
        raise ValueError(f"Unknown benchmark: {bench}")

    env = env_cls(**env_kwargs)
    results = []
    fakg_sizes = []

    n_tasks = min(n_episodes, len(env) if hasattr(env, "__len__") else n_episodes)

    for ep_idx in range(n_tasks):
        print(f"  Episode {ep_idx+1}/{n_tasks}", end="\r")
        try:
            if bench == "intercode":
                obs = env.reset(idx=ep_idx)
            elif bench == "webarena":
                obs = env.reset(idx=ep_idx)
            else:
                obs = env.reset()

            task_text = getattr(env, "task_text", obs.split("\n")[0])
            task_emb = embed_fn(task_text)

            class _EnvShim:
                def __init__(self, _env):
                    self._e = _env
                    self.success = False
                def reset(self): return obs
                def step(self, action):
                    result = self._e.step(action)
                    self.success = self._e.success
                    return result

            shim = _EnvShim(env)
            episode = agent.run(task_text, shim)

            new_fpts = miner.mine_episode(agent.trajectory, task_text, env_name, ep_idx)
            for fpt in new_fpts:
                afm.on_add(fpt, ep_idx, task_emb)

            corrections_injected = sum(
                1 for s in episode["trajectory"] if s.get("correction_injected")
            )
            results.append(EpisodeResult(
                success=episode["success"],
                steps=episode["steps"],
                corrections_injected=corrections_injected,
            ))
            fakg_sizes.append(len(fakg))

        except Exception as e:
            print(f"\n  [Episode {ep_idx} error] {e}")
            results.append(EpisodeResult(success=False, steps=0))

    print()
    growth = (fakg_sizes[-1] - fakg_sizes[0]) / max(len(fakg_sizes), 1) if fakg_sizes else 0
    fakg_stats = {**afm.stats(), "growth_rate": growth}
    metrics = compute_metrics(results, fakg_stats)
    return {
        "tsr": metrics.tsr,
        "shr": metrics.shr,
        "gcs": metrics.gcs,
        "cp": metrics.cp,
        "fakg_growth": metrics.fakg_growth_rate,
        "churn": metrics.memory_churn_rate,
        "n_episodes": metrics.n_episodes,
        "str": str(metrics),
    }


def main():
    parser = argparse.ArgumentParser(description="FAILGROUND experiment runner")
    parser.add_argument("--benchmark",
                        choices=["alfworld", "intercode", "webarena", "all"],
                        default="alfworld")
    parser.add_argument("--ablation", default="none",
                        choices=["none", "A1", "A2", "A3", "A4", "A5", "A6"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    create_if_not_exist(args.output)
    agent, miner, afm, fakg, embed_fn = build_system(cfg, args.ablation)

    benchmarks = (
        ["alfworld", "intercode", "webarena"]
        if args.benchmark == "all"
        else [args.benchmark]
    )

    all_results = {}
    for bench in benchmarks:
        print(f"\n=== {bench.upper()} | ablation={args.ablation} | episodes={args.episodes} ===")
        metrics = run_benchmark(bench, agent, miner, afm, fakg, embed_fn, cfg, args.episodes)
        print(f"  {metrics['str']}")
        all_results[bench] = metrics

    out_path = Path(args.output) / f"{args.benchmark}_{args.ablation}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults -> {out_path}")

    fakg_path = Path(args.output) / f"fakg_{args.benchmark}_{args.ablation}.json"
    fakg.save(str(fakg_path))
    print(f"FAKG  -> {fakg_path}  ({len(fakg)} failure nodes)")
    print(f"AFM   -> {afm.stats()}")


if __name__ == "__main__":
    main()
