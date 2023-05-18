import os
from contextlib import contextmanager
from typing import IO

from corpus.word import Word


class Corpus:
    filename: str

    def __init__(self, path):
        backend_dir_path = os.path.abspath(os.path.dirname(__file__))
        self.path = os.path.join(backend_dir_path, path)

    def get_sentence(self, line_num: int) -> list[Word]:
        raise NotImplementedError

    def get_paragraph(self, line_num: int) -> list[list[Word]]:
        raise NotImplementedError

    @contextmanager
    def open_corpus(self) -> IO:
        yield open(file=f'{self.path}/{self.filename}', mode='r+', encoding='utf-8')