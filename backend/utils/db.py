import os
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv

local_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../.env')
if os.path.isfile(local_file_path):
    print(f'Loading environment from {local_file_path}')
    load_dotenv(local_file_path, verbose=True)


@contextmanager
def db_cursor():
    with psycopg.connect(os.getenv('DB_URL')) as conn:
        yield conn.cursor()
