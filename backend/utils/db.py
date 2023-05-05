import os
from contextlib import contextmanager

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


@contextmanager
def db_cursor() -> Cursor | ServerCursor:
    with psycopg.connect(os.getenv('DB_URL')) as conn:
        yield conn.cursor(row_factory=dict_row)
