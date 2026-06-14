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


def extract_entities(text: str) -> list[str]:
    """Extract named entities from text. Returns deduplicated list."""
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        entities = list({ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 1})
        return entities
    # Fallback: capitalized word sequences
    matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    return list(set(matches))


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
        import json
        response = llm_fn(prompt)
        claims = json.loads(response)
        return claims if isinstance(claims, list) else []
    except Exception:
        return []
