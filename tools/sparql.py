"""
SPARQL verification against Wikidata and ConceptNet.

Used by failure step mining to verify atomic claims:
  - If Wikidata SPARQL returns a triple contradicting the action claim -> failure.
  - Resolves entity QIDs for FAKG WorldNode anchoring.

Uses public Wikidata SPARQL endpoint (free).
ConceptNet: local 5.5 dump (assertions.csv).
"""

from __future__ import annotations
import json
import os
import re
import time
from typing import Optional
import requests


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "FAILGROUND/0.1 (research; contact via GitHub)",
}

_QID_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", ".qid_cache.json")
_qid_cache: dict[str, Optional[str]] = {}


def _load_qid_cache():
    global _qid_cache
    try:
        with open(_QID_CACHE_PATH) as f:
            _qid_cache = json.load(f)
    except Exception:
        _qid_cache = {}


def _save_qid_cache():
    try:
        with open(_QID_CACHE_PATH, "w") as f:
            json.dump(_qid_cache, f)
    except Exception:
        pass


_load_qid_cache()
_wikidata_disabled = False  # set True after 403 to skip remaining calls


def entity_to_qid(entity_name: str) -> Optional[str]:
    """Resolve entity name to Wikidata QID. Disk-cached, skips lowercase tokens."""
    global _wikidata_disabled
    if _wikidata_disabled:
        return None
    key = entity_name.lower().strip()
    if key in _qid_cache:
        return _qid_cache[key]
    # skip short tokens and all-lowercase words (SQL keywords, common nouns)
    if len(key) < 2 or entity_name.strip().islower():
        _qid_cache[key] = None
        return None
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": "en",
        "format": "json",
        "limit": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=3)
        if resp.status_code == 403:
            _wikidata_disabled = True
            return None
        resp.raise_for_status()
        results = resp.json().get("search", [])
        qid = results[0]["id"] if results else None
        _qid_cache[key] = qid
        if len(_qid_cache) % 50 == 0:
            _save_qid_cache()
        return qid
    except Exception:
        _qid_cache[key] = None
        return None


def sparql_query(query: str, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            resp = requests.get(
                WIKIDATA_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=WIKIDATA_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def verify_claim(h_qid: str, relation_pid: str, t_label: str) -> tuple[bool, Optional[str]]:
    """
    Check if (h_qid, relation_pid) has a value in Wikidata.
    Returns (contradicts, correct_value_or_None).
    contradicts=True if the stored value != t_label.
    """
    query = f"""
    SELECT ?tailLabel WHERE {{
      wd:{h_qid} wdt:{relation_pid} ?tail .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?tail rdfs:label ?tailLabel. }}
    }}
    LIMIT 1
    """
    result = sparql_query(query)
    if result is None:
        return False, None
    bindings = result.get("results", {}).get("bindings", [])
    if not bindings:
        return False, None
    stored = bindings[0].get("tailLabel", {}).get("value", "")
    contradicts = stored.lower().strip() != t_label.lower().strip()
    return contradicts, stored if contradicts else None


# ConceptNet local lookup
def conceptnet_lookup(entity: str, relation: str, dump_path: str) -> list[tuple[str, str, str]]:
    """
    Search local ConceptNet dump for triples matching entity as head.
    Returns list of (h, r, t).
    """
    results = []
    try:
        with open(dump_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h, r, t = parts[0], parts[1], parts[2]
                if entity.lower() in h.lower() and relation.lower() in r.lower():
                    results.append((h, r, t))
    except FileNotFoundError:
        pass
    return results
