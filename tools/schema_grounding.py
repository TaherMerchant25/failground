"""
Schema-grounded hallucination detection.

Replaces the disabled Wikidata/SPARQL verification path with a deterministic,
reproducible grounding source: the environment's own schema / admissible set.

A claim is a *hallucination* iff it references an identifier that provably does
not exist in the environment's ground truth:

  * InterCode-SQL: an SQL identifier (table / column) not present in the active
    database schema  ->  gamma = "schema_violation".
  * ALFWorld: an object / receptacle referenced by an action that is not in the
    current admissible-object set  ->  gamma = "hallucinated_entity".

No external API, no network, fully deterministic. `correct` is the closest valid
identifier (difflib) so the FPT carries an actionable correction.

env_ctx shapes (produced by env.grounding_context()):
  {"type": "sql",      "schema": {table_name: [col, ...]}}
  {"type": "alfworld", "admissible": [obj_str, ...]}
"""

from __future__ import annotations
import difflib
import re
from typing import Optional

# ----------------------------------------------------------------------------
# SQL
# ----------------------------------------------------------------------------

# Reserved words + function names that must never be treated as schema identifiers.
SQL_STOPWORDS = {
    "select", "from", "where", "join", "inner", "outer", "left", "right", "full",
    "on", "as", "and", "or", "not", "in", "is", "null", "group", "by", "order",
    "having", "limit", "offset", "asc", "desc", "distinct", "count", "sum", "avg",
    "min", "max", "between", "like", "exists", "union", "all", "case", "when",
    "then", "else", "end", "insert", "into", "values", "update", "set", "delete",
    "create", "table", "drop", "alter", "add", "primary", "key", "foreign",
    "references", "default", "true", "false", "intersect", "except", "with",
    "cast", "substr", "substring", "upper", "lower", "length", "abs", "round",
    "now", "date", "year", "month", "day", "t1", "t2", "t3", "t4", "t5",
}

# Tokens like Table.Column, names, aliases.
_SQL_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def build_sql_schema_index(schema: dict[str, list[str]]) -> set[str]:
    """Flatten {table: [cols]} into a lowercased set of every valid identifier."""
    index: set[str] = set()
    for table, cols in (schema or {}).items():
        index.add(table.strip().lower())
        for c in cols:
            index.add(c.strip().lower())
    return index


def extract_sql_identifiers(sql: str) -> list[str]:
    """
    Pull candidate schema identifiers from raw SQL.

    Strips string literals, splits Table.Column into both parts, drops pure
    numbers and SQL keywords/aliases. Returns lowercased identifiers in order
    (deduplicated, order-preserving).
    """
    # remove quoted string literals so their contents aren't treated as identifiers
    sql_nostr = re.sub(r"'[^']*'", " ", sql)
    sql_nostr = re.sub(r'"[^"]*"', " ", sql_nostr)
    seen: set[str] = set()
    out: list[str] = []
    # split dotted refs so "t1.name" -> "t1", "name"
    for raw in re.split(r"[^\w.]+", sql_nostr):
        for part in raw.split("."):
            tok = part.strip().lower()
            if not tok or tok in SQL_STOPWORDS:
                continue
            if not _SQL_TOKEN.fullmatch(tok):
                continue
            if tok.isdigit():
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
    return out


_TABLE_ALIAS = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)
_COL_ALIAS = re.compile(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE)


def extract_declared_aliases(sql: str) -> set[str]:
    """Table aliases (FROM/JOIN x AS y) and computed-column aliases (AS y)
    are newly-introduced names, not schema lookups — exempt them."""
    aliases: set[str] = set()
    for m in _TABLE_ALIAS.finditer(sql):
        alias = m.group(2).lower()
        if alias not in SQL_STOPWORDS:
            aliases.add(alias)
    for m in _COL_ALIAS.finditer(sql):
        aliases.add(m.group(1).lower())
    return aliases


def sql_hallucinations(sql: str, schema: dict[str, list[str]]) -> list[dict]:
    """
    Identifiers in `sql` that are absent from the schema = schema_violation
    hallucinations. Returns list of records:
      {h, r, t, gamma, correct}
    where h = offending identifier, correct = closest valid schema id (or "").
    """
    index = build_sql_schema_index(schema)
    if not index:
        return []
    aliases = extract_declared_aliases(sql)
    bad: list[dict] = []
    for ident in extract_sql_identifiers(sql):
        if ident in index or ident in aliases:
            continue
        match = difflib.get_close_matches(ident, index, n=1, cutoff=0.6)
        bad.append({
            "h": ident,
            "r": "not_in_schema",
            "t": "schema",
            "gamma": "schema_violation",
            "correct": match[0] if match else "",
        })
    return bad


# ----------------------------------------------------------------------------
# ALFWorld
# ----------------------------------------------------------------------------

# Verbs / function words that are not objects.
ALF_STOPWORDS = {
    "go", "to", "the", "a", "an", "take", "from", "put", "in", "on", "open",
    "close", "toggle", "use", "heat", "cool", "clean", "slice", "look", "move",
    "examine", "with", "and", "of", "into", "at", "your", "you",
}


def _object_tokens(text: str) -> list[str]:
    """Extract candidate object words (multi-word objects collapse to head noun)."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in re.split(r"[^\w]+", text.lower()):
        if not tok or tok in ALF_STOPWORDS or tok.isdigit():
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def alfworld_hallucinations(action: str, admissible: list[str]) -> list[dict]:
    """
    Object/receptacle words in `action` that never appear in any admissible
    command = hallucinated_entity. Numbers (instance ids) are ignored so
    "apple 2" vs "apple 1" is not flagged as long as the object type exists.
    """
    if not admissible:
        return []
    admissible_blob = " ".join(admissible).lower()
    admissible_vocab = set(_object_tokens(admissible_blob))
    if not admissible_vocab:
        return []
    bad: list[dict] = []
    for tok in _object_tokens(action):
        if tok in admissible_vocab:
            continue
        match = difflib.get_close_matches(tok, admissible_vocab, n=1, cutoff=0.7)
        bad.append({
            "h": tok,
            "r": "not_admissible",
            "t": "environment",
            "gamma": "hallucinated_entity",
            "correct": match[0] if match else "",
        })
    return bad


# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

def verify_step(action_text: str, env_ctx: Optional[dict]) -> list[dict]:
    """
    Deterministically detect hallucinated identifiers in a raw action given the
    environment grounding context. Returns a (possibly empty) list of FPT-shaped
    hallucination records. Empty list => step is grounded (or not adjudicable).
    """
    if not env_ctx:
        return []
    kind = env_ctx.get("type")
    if kind == "sql":
        return sql_hallucinations(action_text, env_ctx.get("schema", {}))
    if kind == "alfworld":
        return alfworld_hallucinations(action_text, env_ctx.get("admissible", []))
    return []


def is_adjudicable(env_ctx: Optional[dict]) -> bool:
    """Whether the grounding source can decide claims at all (for GCS denominator)."""
    if not env_ctx:
        return False
    if env_ctx.get("type") == "sql":
        return bool(env_ctx.get("schema"))
    if env_ctx.get("type") == "alfworld":
        return bool(env_ctx.get("admissible"))
    return False
