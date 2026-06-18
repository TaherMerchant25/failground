"""
Tripartite Failure-Aware Knowledge Graph (FAKG).

Node types:
  WorldNode   — Wikidata QID or ConceptNet URI (permanent anchor)
  FailureNode — one FPT per confirmed failure
  TaskNode    — task embedding + env label

Edge types:
  WorldEdge           — standard KG relation between world nodes
  CAUSED_BY           — FailureNode -> WorldNode(h)
  CORRECTED_BY        — FailureNode -> WorldNode(a_correct)
  EXPERIENCED         — TaskNode -> FailureNode
"""

from __future__ import annotations
import json
import os
from typing import Optional
import networkx as nx

from fakg.fpt import FPT
from tools.entity_linking import normalize_entity


class FAKG:
    def __init__(self, graph_path: Optional[str] = None):
        self.G: nx.DiGraph = nx.DiGraph()
        self._fpt_index: dict[str, FPT] = {}
        self._hr_attempt: dict[tuple[str, str], int] = {}
        self._hr_fail: dict[tuple[str, str], int] = {}

        if graph_path and os.path.exists(graph_path):
            self.load(graph_path)

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    def _add_world_node(self, entity: str, qid: Optional[str] = None, uri: Optional[str] = None) -> str:
        entity = normalize_entity(entity)
        if not self.G.has_node(entity):
            self.G.add_node(entity, node_type="WorldNode", qid=qid, uri=uri)
        return entity

    def _add_task_node(self, task_id: str, embedding: list[float], env: str):
        if not self.G.has_node(task_id):
            self.G.add_node(task_id, node_type="TaskNode", embedding=embedding, env=env)

    # ------------------------------------------------------------------
    # Core: add failure
    # ------------------------------------------------------------------

    def add_failure(
        self,
        fpt: FPT,
        task_id: str,
        task_embedding: list[float],
        env: str,
        mention_entities: Optional[list[str]] = None,
    ):
        """Insert an FPT into the graph, wiring all cross-layer edges."""
        h_key = self._add_world_node(fpt.h, qid=fpt.wikidata_qid_h, uri=fpt.conceptnet_uri_h)
        correct_key = self._add_world_node(fpt.a_correct, qid=fpt.wikidata_qid_correct)
        self._add_task_node(task_id, task_embedding, env)

        fn_id = fpt.fpt_id
        self.G.add_node(
            fn_id,
            node_type="FailureNode",
            fpt=fpt.to_dict(),
            is_active=True,
            priority=fpt.priority,
        )

        self.G.add_edge(fn_id, h_key, edge_type="CAUSED_BY")
        self.G.add_edge(fn_id, correct_key, edge_type="CORRECTED_BY")
        self.G.add_edge(task_id, fn_id, edge_type="EXPERIENCED")

        # Index action-level noun-chunk entities so CCSR lookup finds this node
        for me in (mention_entities or []):
            me_key = self._add_world_node(me)
            if me_key != h_key:
                self.G.add_edge(fn_id, me_key, edge_type="MENTIONS")

        self._fpt_index[fn_id] = fpt

        key = (fpt.h, fpt.r)
        self._hr_fail[key] = self._hr_fail.get(key, 0) + 1

    def record_attempt(self, h: str, r: str):
        key = (normalize_entity(h), r)
        self._hr_attempt[key] = self._hr_attempt.get(key, 0) + 1

    def get_confidence(self, h: str, r: str) -> float:
        key = (normalize_entity(h), r)
        attempts = self._hr_attempt.get(key, 0)
        fails = self._hr_fail.get(key, 0)
        if attempts == 0:
            return 0.0
        return fails / attempts

    # ------------------------------------------------------------------
    # World-edge (standard KG relation)
    # ------------------------------------------------------------------

    def add_world_edge(self, h: str, r: str, t: str):
        self._add_world_node(h)
        self._add_world_node(t)
        self.G.add_edge(h, t, edge_type="WorldEdge", relation=r)

    # ------------------------------------------------------------------
    # Active failure nodes
    # ------------------------------------------------------------------

    def active_failure_nodes(self) -> list[FPT]:
        return [
            self._fpt_index[n]
            for n, d in self.G.nodes(data=True)
            if d.get("node_type") == "FailureNode" and d.get("is_active", False)
        ]

    def failure_nodes_for_entity(self, entity: str) -> list[FPT]:
        """All active FailureNodes with CAUSED_BY or MENTIONS edge -> entity."""
        entity = normalize_entity(entity)
        result = []
        if not self.G.has_node(entity):
            return result
        seen: set[str] = set()
        for pred in self.G.predecessors(entity):
            d = self.G.nodes[pred]
            if d.get("node_type") == "FailureNode" and d.get("is_active", False):
                edge_type = self.G.edges[pred, entity].get("edge_type", "")
                if edge_type in ("CAUSED_BY", "MENTIONS") and pred not in seen:
                    seen.add(pred)
                    result.append(self._fpt_index[pred])
        return result

    def flat_lookup(self, entity: str) -> list[FPT]:
        """
        A2 ablation: no provenance-edge traversal. Matches only the bare
        claim subject (h), ignoring MENTIONS/action-context entities and
        the CAUSED_BY/CORRECTED_BY graph structure entirely.
        """
        entity = normalize_entity(entity)
        return [
            fpt for fpt in self._fpt_index.values()
            if normalize_entity(fpt.h) == entity and fpt.is_active
        ]

    # ------------------------------------------------------------------
    # Ghost state
    # ------------------------------------------------------------------

    def deactivate(self, fpt_id: str):
        if self.G.has_node(fpt_id):
            self.G.nodes[fpt_id]["is_active"] = False
            if fpt_id in self._fpt_index:
                self._fpt_index[fpt_id].is_active = False

    def reactivate(self, h: str, r: str):
        """Reactivate ghost nodes matching (h, r) to prevent cold-start."""
        for fpt_id, fpt in self._fpt_index.items():
            if fpt.h == h and fpt.r == r and not fpt.is_active:
                self.G.nodes[fpt_id]["is_active"] = True
                fpt.is_active = True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        data = nx.node_link_data(self.G)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.G = nx.node_link_graph(data)
        for n, d in self.G.nodes(data=True):
            if d.get("node_type") == "FailureNode":
                fpt = FPT.from_dict(d["fpt"])
                self._fpt_index[n] = fpt

    def __len__(self):
        return sum(
            1 for _, d in self.G.nodes(data=True) if d.get("node_type") == "FailureNode"
        )
