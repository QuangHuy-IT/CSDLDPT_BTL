from pathlib import Path

from database.db import get_conn


def main():
    schema_path = Path(__file__).resolve().parents[1] / "database" / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")

    conn = get_conn()
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql)

    print("Database initialized.")


if __name__ == "__main__":
    main()
