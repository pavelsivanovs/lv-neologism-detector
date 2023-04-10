import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO


@dataclass
class Word:
    word: str
    lemma: str


def get_text_sentence(sent: list[Word]) -> str:
    txt = ''
    for word in sent:
        txt += word.word + ' '
    return txt


class CommonCrawl:
    filename = 'commoncrawl.vert'

    def __init__(self, path):
        self.path = path

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
    corpus = CommonCrawl('../corpus')
    sentence = corpus.get_sentence(44967)
    print(get_text_sentence(sentence))
