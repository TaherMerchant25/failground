"""
One-off converter: InterCode-Spider MySQL dump -> per-database sqlite files.

No live MySQL server available, so this does regex-based syntax translation
(strip MySQL-only clauses, backticks, AUTO_INCREMENT) rather than a real
mysqldump replay. Good enough for read-only SELECT execution-accuracy checks.
"""

from __future__ import annotations
import re
import os
import sqlite3

DUMP_PATH = os.path.join(
    os.path.dirname(__file__),
    "../benchmarks/intercode/data/sql/spider/ic_spider_dbs.sql",
)
OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "../benchmarks/intercode/data/sql/spider/dbs",
)


def clean_create_table(stmt: str) -> str:
    stmt = re.sub(r"/\*!.*?\*/", "", stmt, flags=re.DOTALL)
    stmt = stmt.replace("`", '"')
    # strip MySQL index-prefix-length specs: "col"(255) -> "col"
    stmt = re.sub(r'("\w+")\(\d+\)', r"\1", stmt)
    stmt = re.sub(r"\bENGINE=\w+\b", "", stmt)
    stmt = re.sub(r"\bDEFAULT CHARSET=\w+\b", "", stmt)
    stmt = re.sub(r"\bAUTO_INCREMENT=\d+\b", "", stmt)
    stmt = re.sub(r"\bCOLLATE[= ]\S+\b", "", stmt)
    stmt = re.sub(r"\bCHARACTER SET \S+\b", "", stmt)
    stmt = re.sub(r"\bUNSIGNED\b", "", stmt, flags=re.IGNORECASE)
    stmt = re.sub(r"\bAUTO_INCREMENT\b", "", stmt, flags=re.IGNORECASE)
    # strip MySQL-only secondary index defs: "[UNIQUE] KEY name (cols)," (keep PRIMARY/FOREIGN KEY)
    stmt = re.sub(
        r"(?<!PRIMARY )(?<!FOREIGN )\b(?:UNIQUE\s+)?KEY\s+\S+\s*\([^)]*\),?\s*",
        "", stmt, flags=re.IGNORECASE,
    )
    stmt = re.sub(r"\bint\(\d+\)", "INTEGER", stmt, flags=re.IGNORECASE)
    stmt = re.sub(r"\bint\b", "INTEGER", stmt, flags=re.IGNORECASE)
    stmt = re.sub(r"\bvarchar\(\d+\)", "TEXT", stmt, flags=re.IGNORECASE)
    stmt = re.sub(r"\bdatetime\b", "TEXT", stmt, flags=re.IGNORECASE)
    stmt = re.sub(r",\s*\)", ")", stmt)
    return stmt


def clean_insert(stmt: str) -> str:
    stmt = stmt.replace("`", '"')
    # MySQL escapes embedded quotes as \' ; SQLite wants ''
    stmt = stmt.replace("\\'", "''")
    return stmt


def split_statements(sql_block: str) -> list[str]:
    statements = []
    buf = []
    in_string = False
    quote_char = None
    i = 0
    while i < len(sql_block):
        ch = sql_block[i]
        if in_string:
            buf.append(ch)
            if ch == "\\":
                if i + 1 < len(sql_block):
                    buf.append(sql_block[i + 1])
                    i += 2
                    continue
            elif ch == quote_char:
                in_string = False
        else:
            if ch in ("'", '"'):
                in_string = True
                quote_char = ch
                buf.append(ch)
            elif ch == ";":
                statements.append("".join(buf))
                buf = []
                i += 1
                continue
            else:
                buf.append(ch)
        i += 1
    if buf and "".join(buf).strip():
        statements.append("".join(buf))
    return statements


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DUMP_PATH, "r", encoding="utf-8") as f:
        dump = f.read()

    db_sections = re.split(r"CREATE DATABASE\s+IF NOT EXISTS\s*", dump)[1:]

    n_dbs = 0
    n_tables = 0
    for section in db_sections:
        m = re.match(r"`?(\w+)`?", section)
        if not m:
            continue
        db_name = m.group(1)
        out_path = os.path.join(OUT_DIR, f"{db_name}.sqlite")
        if os.path.exists(out_path):
            os.remove(out_path)
        conn = sqlite3.connect(out_path)
        cur = conn.cursor()

        for stmt in split_statements(section):
            stripped = stmt.strip()
            if not stripped:
                continue
            upper = stripped.upper()
            if upper.startswith("CREATE TABLE"):
                clean = clean_create_table(stmt)
                try:
                    cur.execute(clean)
                    n_tables += 1
                except sqlite3.OperationalError as e:
                    print(f"  [warn] {db_name}: CREATE TABLE failed: {e}")
            elif upper.startswith("INSERT INTO"):
                clean = clean_insert(stmt)
                try:
                    cur.execute(clean)
                except sqlite3.OperationalError as e:
                    print(f"  [warn] {db_name}: INSERT failed: {e}")
            # skip DROP TABLE, LOCK/UNLOCK TABLES, ALTER TABLE ... KEYS, SET, etc.

        conn.commit()
        conn.close()
        n_dbs += 1
        print(f"[mysql_to_sqlite] {db_name} -> {out_path}")

    print(f"\nConverted {n_dbs} databases, {n_tables} tables -> {OUT_DIR}")


if __name__ == "__main__":
    main()
