import re

from corpus.init_corpus import Corpus
from corpus.word import Word, get_text_sentence


class CommonCrawl(Corpus):
    """ Corpus class for interacting with CommonCrawl corpus, which stores Tīmeklis2020 data. """
    filename = 'commoncrawl.vert'

    def __init__(self, path='../../corpus'):
        super().__init__(path)

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

            corpus.seek(sentence_start_pos)  # returns to the first word of a sentence
            while '</s>' not in (line := corpus.readline()):
                if line[0] == '<':
                    continue
                line = re.split(r'[\t\n]', line)
                word = Word(word=line[0], lemma=line[2])
                sentence.append(word)
            return sentence


if __name__ == '__main__':
    corpus = CommonCrawl()
    sentence = corpus.get_sentence(44967)
    print(get_text_sentence(sentence))
