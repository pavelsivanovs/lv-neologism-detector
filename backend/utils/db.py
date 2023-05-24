import os
from contextlib import contextmanager
from functools import cache

import psycopg
from dotenv import load_dotenv
from psycopg import Cursor, ServerCursor
from psycopg.rows import dict_row

from corpus.word import Word
from utils.logger import get_configured_logger

logger = get_configured_logger(__name__)

local_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../.env')
if os.path.isfile(local_file_path):
    logger.info(f'Loading environment from {local_file_path}')
    load_dotenv(local_file_path, verbose=True)


def get_similar_words(word: str, limit=5) -> list[dict]:
    word = word.lower()
    with db_cursor() as cur:
        cur.execute('''
            with words_with_distance as (
                select 
                    l.lemma as word,
                    levenshtein(%s, substr(lower(l.lemma), 1, 200)) as distance,
                    similarity(%s, l.lemma) as similarity
                from dict.lemmas l
            )
            select * from words_with_distance wwd
            order by wwd.distance
            limit %s;
        ''', (word, word, limit))
        return cur.fetchall()


@cache
def is_word_in_db(word: str) -> bool:
    """ Checks whether word is present among dictionary entries. """
    if isinstance(word, Word):
        word = word.lemma
    word = word.lower()
    with db_cursor() as cur:
        q = 'select lemma from dict.lemmas where lemma = %s'
        cur.execute(q, (word,))
        return bool(cur.fetchone())


@contextmanager
def db_cursor() -> Cursor | ServerCursor:
    with psycopg.connect(os.getenv('DB_URL')) as conn:
        yield conn.cursor(row_factory=dict_row)
