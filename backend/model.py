import csv
import re
import time

from commoncrawl import CommonCrawl, Word
from stopwords import get_stopwords
from utils.db import db_cursor
from utils.logger import get_configured_logger

logger = get_configured_logger(__name__)


def is_word_in_db(word: str) -> bool:
    """ Checks whether word is present among dictionary entries. """
    with db_cursor() as cur:
        q = 'select id from dict.entries where lower(heading) = %s'
        cur.execute(q, (word,))
        return bool(cur.fetchone())


def is_neologism(word: str) -> float:
    """ Returns probability of a word being a real neologism, """
    pass


def normalize_lemma(word: Word):
    if word.tagset[0] == 'v':
        # removing "ne-" from lemma
        if word.tagset[-1:] == 'y':
            word.lemma = word.lemma[2:]

    # removing "-šana" and substituting it with "-t"
    word.lemma = re.sub(r'šana$', 't', word.lemma)


def write_non_existing_lemmas_to_csv():
    corpus = CommonCrawl()
    stopwords = get_stopwords()

    logger.info('starting iterating the corpus')
    start_time = time.time()

    with corpus.open_corpus() as corpus_file, \
            open('lemma-not-found-in-tezaurs_16_04.csv', 'w', encoding='utf-8', newline='') as output:
        fieldnames = ['number_line', 'word', 'supposed_lemma', 'tagset']
        writer = csv.DictWriter(output, fieldnames)
        writer.writeheader()

        written_counter = 0
        for idx, line in enumerate(corpus_file, start=1):
            if idx <= 1644955:
                continue

            if idx % 10000 == 0:
                logger.info(f'iterating line {idx}')

            if line[0] == '<' or line[0] in stopwords:
                continue

            # TODO
            #  - adverbs mostly are not present in thesaurus, thus gotta
            #    transfer them to adjective and then look up in thesaurus

            line = re.split(r'[\t\n]', line)

            if line[1][0] == 'z' or is_word_in_db(line[2].lower()):
                continue

            written_counter += 1
            logger.info(f'writing into CSV file line: {line}')
            writer.writerow({
                'number_line': idx,
                'tagset': line[1],
                'word': line[0],
                'supposed_lemma': line[2],
            })

    logger.info('finished iterating over corpus')
    end_time = time.time()
    elapsed_time = time.gmtime(end_time - start_time)
    logger.info(f'Elapsed time: {elapsed_time}')
    logger.info(f'Words written: {written_counter}')


def model():
    """  """
    pass


# TODO
#  1. check whether there is index by words in entries table
#  1. get the text
#  2. move it through LVTagger or sth
#  3. check if words are in DB
#  4. ML classification


if __name__ == '__main__':
    write_non_existing_lemmas_to_csv()
