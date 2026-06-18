"""
Entity linking: spaCy NER + optional REL linking to Wikidata QIDs.

Used by CCSR to extract entities E from action text before FAKG lookup.
Falls back to simple noun-phrase extraction if REL is unavailable.
"""

from __future__ import annotations
from typing import Optional
import re

try:
    import spacy
    _nlp = None
    def _get_nlp():
        global _nlp
        if _nlp is None:
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                _nlp = None
        return _nlp
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    def _get_nlp(): return None


def normalize_entity(s: str) -> str:
    """Lowercase + strip for consistent FAKG store/lookup."""
    return s.strip().lower()


def extract_entities(text: str) -> list[str]:
    """Extract named entities and noun chunks from text. Returns deduplicated list."""
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        entities = {ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 1}
        # also include noun chunk roots (covers common nouns like "apple", "table", SQL table names)
        for chunk in doc.noun_chunks:
            t = chunk.root.lemma_.strip().lower()
            if len(t) > 2:
                entities.add(t)
        return list(entities)
    # Fallback: all words longer than 3 chars
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
    return list({w.lower() for w in words})


def extract_atomic_claims(action_text: str, llm_fn) -> list[dict]:
    """
    FActScore-style decomposition: split action into atomic (h, r, t) claims.
    llm_fn(prompt) -> str.
    """
    prompt = (
        f"Decompose the following agent action into atomic factual claims.\n"
        f"Each claim should be a (subject, predicate, object) triple.\n"
        f"Action: {action_text}\n\n"
        f"Output as JSON array: [{{'h': '...', 'r': '...', 't': '...'}}]"
    )
    try:
        import json, re
        response = llm_fn(prompt)
        # strip markdown code fences (LLM often wraps with ```json ... ```)
        response = re.sub(r'^```(?:json)?\s*', '', response.strip())
        response = re.sub(r'\s*```$', '', response.strip())
        claims = json.loads(response)
        return claims if isinstance(claims, list) else []
    except Exception:
        return []
