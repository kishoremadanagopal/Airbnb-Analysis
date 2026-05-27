"""
Load airbnb_clean.csv into SQLite and run the analytical queries
defined in 02_sql_analysis.sql, printing results to stdout.

Usage:
    python 02_sql_demo.py
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "airbnb.db"
CSV_PATH = Path(__file__).parent / "airbnb_clean.csv"
SQL_PATH = Path(__file__).parent / "02_sql_analysis.sql"


def build_database() -> sqlite3.Connection:
    """Load the cleaned CSV into a fresh SQLite database."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH.name} not found. Run the notebook 01_analysis.ipynb first."
        )
    if DB_PATH.exists():
        DB_PATH.unlink()

    df = pd.read_csv(CSV_PATH)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("listings", conn, index=False)

    # Index the most-queried columns
    conn.executescript("""
        CREATE INDEX idx_borough  ON listings(neighbourhood_group);
        CREATE INDEX idx_room     ON listings(room_type);
        CREATE INDEX idx_host     ON listings(host_id);
    """)
    conn.commit()
    return conn


def parse_queries(sql_text: str) -> list[tuple[str, str]]:
    """
    Split the SQL file into (label, query) pairs.
    A query block starts with a header line like:  -- Q1. Borough summary
    and ends at the next such header or end of file.
    """
    header_re = re.compile(r"^--\s*(Q\d+\..*)$", re.MULTILINE)
    matches = list(header_re.finditer(sql_text))
    pairs = []
    for i, m in enumerate(matches):
        label = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(sql_text)
        body = sql_text[start:end]
        # Strip comment-only lines and dash separators
        sql_only = "\n".join(
            ln for ln in body.splitlines()
            if not ln.strip().startswith("--") and ln.strip()
        ).strip()
        if sql_only:
            pairs.append((label, sql_only))
    return pairs


def run(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    return pd.read_sql(query, conn)


def main() -> None:
    print("=" * 78)
    print("NYC AIRBNB — SQL ANALYSIS")
    print("=" * 78)

    conn = build_database()
    print(f"\nLoaded into SQLite: {DB_PATH.name}")
    n = run(conn, "SELECT COUNT(*) AS n FROM listings").iloc[0, 0]
    print(f"Listings table: {n:,} rows\n")

    sql_text = SQL_PATH.read_text()
    queries = parse_queries(sql_text)
    print(f"Running {len(queries)} analytical queries...\n")

    for label, query in queries:
        print("-" * 78)
        print(label)
        print("-" * 78)
        try:
            result = run(conn, query)
            # Print a tidy table
            with pd.option_context(
                "display.max_columns", 20,
                "display.width", 160,
                "display.float_format", "{:.2f}".format,
            ):
                print(result.to_string(index=False))
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    conn.close()
    print("=" * 78)
    print("Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
