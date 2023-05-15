import os
from contextlib import contextmanager
from functools import cache

import psycopg
from dotenv import load_dotenv
from psycopg import Cursor, ServerCursor
from psycopg.rows import dict_row

from utils.logger import get_configured_logger

logger = get_configured_logger(__name__)

local_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../.env')
if os.path.isfile(local_file_path):
    logger.info(f'Loading environment from {local_file_path}')
    load_dotenv(local_file_path, verbose=True)


@cache
def is_word_in_db(word: str) -> bool:
    """ Checks whether word is present among dictionary entries. """
    with db_cursor() as cur:
        q = 'select id from dict.entries where lower(heading) = %s'
        cur.execute(q, (word,))
        return bool(cur.fetchone())


@contextmanager
def db_cursor() -> Cursor | ServerCursor:
    with psycopg.connect(os.getenv('DB_URL')) as conn:
        yield conn.cursor(row_factory=dict_row)
