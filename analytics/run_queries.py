import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = "database/ecommerce.db"
QUERIES_FILE = "analytics/queries.sql"


def load_queries(filepath: str) -> list:
    """Split a .sql file into individual queries (separated by ';')."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove SQL comments
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        lines.append(line)
    content = "\n".join(lines)

    queries = [q.strip() for q in content.split(";") if q.strip()]
    return queries


def run_all_queries(db_path: str = DB_PATH, queries_file: str = QUERIES_FILE):
    conn = sqlite3.connect(db_path)
    queries = load_queries(queries_file)

    print(f"[INFO] Running {len(queries)} queries...\n")

    for i, query in enumerate(queries, 1):
        # Get first non-empty meaningful line as title
        first_line = next(
            (l.strip() for l in query.splitlines() if l.strip() and not l.strip().startswith("--")),
            f"Query {i}"
        )
        print("=" * 60)
        print(f"[{i}] {first_line[:80]}")
        print("=" * 60)

        try:
            df = pd.read_sql_query(query, conn)
            print(df.to_string(index=False))
            print(f"\n({len(df)} rows)\n")
        except Exception as e:
            print(f"[ERROR] {e}\n")

    conn.close()


def run_single(query: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Run a single query and return a DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


if __name__ == "__main__":
    run_all_queries()