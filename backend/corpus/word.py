import dataclasses
from dataclasses import dataclass


@dataclass
class Word:
    word: str
    lemma: str
    tagset: str = None
    sentence: str = None

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

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
