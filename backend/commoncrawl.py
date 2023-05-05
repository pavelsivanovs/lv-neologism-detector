import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO

from utils.db import db_cursor


@dataclass
class Word:
    word: str
    lemma: str
    tagset: str = None

    @staticmethod
    def get_word_from_line(line):
        if line == '':
            return
        line = line.split('\t')
        return Word(word=line[0], lemma=line[2], tagset=line[1])


def get_text_sentence(sent: list[Word]) -> str:
    txt = ''
    for word in sent:
        txt += word.word + ' '
    return txt


def get_similar_words(word: str, limit=5) -> list[dict]:
    word = word.lower()
    with db_cursor() as cur:
        cur.execute(f'''
            with words_with_distance as (
                select e.id, e.heading as word, levenshtein('{word}', substr(lower(e.heading), 1, 200)) as distance
                from dict.entries e
            )
            select * from words_with_distance wwd
            order by wwd.distance
            limit {limit};
        ''')
        return cur.fetchall()


class CommonCrawl:
    filename = 'commoncrawl.vert'

    def __init__(self, path='../corpus'):
        backend_dir_path = os.path.abspath(os.path.dirname(__file__))
        self.path = os.path.join(backend_dir_path, path)

    def get_sentence(self, line_num: int) -> list[Word]:
        sentence = []
        with self.open_corpus() as corpus:
            current_pos = 0
            sentence_start_pos = 0
            for i, line in enumerate(corpus, start=1):
                if i == line_num:
                    break
                current_pos += len(line.encode('utf-8'))
                if '<s>' in line:
                    sentence_start_pos = current_pos

            corpus.seek(sentence_start_pos) # returns to the first word of a sentence
            while '</s>' not in (line := corpus.readline()):
                if line[0] == '<':
                    continue
                line = re.split(r'[\t\n]', line)
                word = Word(word=line[0], lemma=line[2])
                sentence.append(word)
            return sentence

    def get_paragraph(self, line_num: int) -> list[list[Word]]:
        pass

    @contextmanager
    def open_corpus(self) -> IO:
        yield open(file=f'{self.path}/{self.filename}', mode='r+', encoding='utf-8')


if __name__ == '__main__':
    corpus = CommonCrawl()
    sentence = corpus.get_sentence(44967)
    print(get_text_sentence(sentence))
