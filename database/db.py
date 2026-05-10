import os

import psycopg2


def get_conn():
    url = os.getenv("DATABASE_URL")
    if url:
        return psycopg2.connect(url)

    host = os.getenv("PGHOST", "localhost")
    port = int(os.getenv("PGPORT", "5432"))
    dbname = os.getenv("PGDATABASE", "audio_search")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )


def fetch_all_features(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.audio_file_id,
                   f.feature_vector,
                   f.f0_median,
                   a.file_path,
                   a.name_intrument,
                   a.note,
                   a.octave,
                   a.dynamic
            FROM features f
            JOIN audiofiles a ON a.id = f.audio_file_id
            """
        )
        return cur.fetchall()
