# FAILGROUND
**Failure-Provenance Knowledge Graphs for Step-Level Grounding of LLM Agent Reasoning**

*AKBC 2026 Workshop — Short Research Paper (4–6 pages)*

---

## Overview

FAILGROUND unifies two independent lines of work:
- **FAILKG** — mines agent failure trajectories into structured (h,r,t) triples (but orphaned, no world anchor)
- **GroundCheck** — validates reasoning steps via SPARQL against Wikidata/ConceptNet (but no failure memory)

Novel contributions:
1. **FPT schema** — 8-tuple `(h, r, t, γ, θ_task, t_ep, c, a_correct)` with outcome-validated confidence
2. **Tripartite FAKG** — WorldNode / FailureNode / TaskNode with typed cross-layer edges
3. **CCSR** — Coverage-Complete Subgraph Retrieval guarantees every entity in an action is checked
4. **AFM** — Adaptive Failure Memory with composite priority pruning + ghost-state eviction

**Total compute cost: <$12 (GPT-4o-mini inference only)**

---

## Structure

```
failground/
├── fakg/              # Core system
│   ├── fpt.py         # Failure-Provenance Triple dataclass
│   ├── fakg.py        # Tripartite graph (networkx)
│   ├── ccsr.py        # Coverage-Complete Subgraph Retrieval
│   ├── afm.py         # Adaptive Failure Memory + pruning
│   └── mast.py        # MAST failure-mode classifier (14 modes)
├── pipeline/
│   ├── failure_miner.py   # Post-episode failure step mining
│   ├── triple.py          # Triple extraction (from LeanRAG)
│   └── llm_infer.py       # LLM inference wrapper
├── agent/
│   └── react_agent.py     # ReAct + FAILGROUND correction loop
├── tools/
│   ├── sparql.py          # Wikidata SPARQL + ConceptNet lookup
│   └── entity_linking.py  # spaCy NER entity extractor
├── eval/
│   └── metrics.py         # TSR, SHR, GCS, CP, growth/churn rates
├── benchmarks/            # Cloned repos
│   ├── alfworld/
│   ├── intercode/
│   ├── reflexion/         # Baseline
│   └── visualwebarena/
├── papers/                # All 12 literature PDFs
├── run_experiment.py      # Main runner
└── config.yaml            # All hyperparameters
```

---

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

export OPENAI_API_KEY=your_key

# Run ALFWorld baseline
python run_experiment.py --benchmark alfworld --ablation none --episodes 50

# Ablation A4: FIFO vs composite priority
python run_experiment.py --benchmark alfworld --ablation A4 --episodes 50
```

---

## Ablations

| ID | Change | Tests |
|----|--------|-------|
| A1 | FPT → standard (h,r,t) | 8-tuple provenance necessary? |
| A2 | Remove CAUSED_BY/CORRECTED_BY | World-grounding necessary? |
| A3 | CCSR → BM25 top-3 | Coverage guarantee vs top-k? |
| A4 | AFM → FIFO eviction | Composite priority vs FIFO? |
| A5 | γ=0 in AFM | Task-similarity term value? |
| A6 | Remove γ-mode filter | Typed vs untyped retrieval? |

---

## Key References

- ReAct: Yao et al. ICLR 2023 — `papers/react_yao2023.pdf`
- Reflexion: Shinn et al. NeurIPS 2023 — `papers/reflexion_shinn2023.pdf`
- MAST: Cemri et al. NeurIPS 2025 — `papers/mast_cemri2025.pdf`
- FActScore: Min et al. EMNLP 2023 — `papers/factscore_min2023.pdf`
- HippoRAG: Gutierrez et al. NeurIPS 2024 — `papers/hipporag_gutierrez2024.pdf`
- ALFWorld: Shridhar et al. ICLR 2021 — `papers/alfworld_shridhar2021.pdf`
- InterCode: Yang et al. NeurIPS 2023 — `papers/intercode_yang2023.pdf`
