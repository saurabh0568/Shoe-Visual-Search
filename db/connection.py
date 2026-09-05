from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from config import DATABASE_URL


@contextmanager
def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_dict_connection():
    """Same as get_connection, but conn.cursor() returns dict-like rows by default."""
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()
