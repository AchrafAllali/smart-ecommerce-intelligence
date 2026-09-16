import sqlite3
import os

DB_PATH = "database/ecommerce.db"


def load_to_sqlite(data_dict: dict, db_path: str = DB_PATH):
    """Load all tables into SQLite."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        for name, df in data_dict.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
            print(f"[OK] {name}: {len(df):,} rows -> SQLite")
        print(f"\n[INFO] Database location: {db_path}")
    finally:
        conn.close()


def test_connection(db_path: str = DB_PATH):
    """Quick check of the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("[INFO] Tables in database:")
    for t in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {t[0]}")
        count = cursor.fetchone()[0]
        print(f"   - {t[0]}: {count:,} rows")
    conn.close()