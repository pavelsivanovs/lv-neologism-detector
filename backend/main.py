import csv
import json
import re
import time

from progiter import ProgIter

from corpus.commoncrawl import CommonCrawl
from corpus.word import Word
from lemmatizer import Lemmatizer
from utils.db import db_cursor, is_word_in_db
from utils.logger import get_configured_logger
from utils.stopwords import get_stopwords

logger = get_configured_logger(__name__)


def is_neologism(word: str) -> float:
    """ Returns probability of a word being a real neologism, """
    pass


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
            if idx <= 1644955:  # update value to the line last read in the corpus
                continue

            if idx % 10000 == 0:
                logger.info(f'iterating line {idx}')

            if line[0] == '<' or line[0] in stopwords:
                continue

            # TODO
            #  - adverbs mostly are not present in thesaurus, thus gotta
            #    transfer them to adjective and then look up in thesaurus

            line = re.split(r'[\t\n]', line)

            if line[1][0] == 'z' or is_word_in_db(line[2]):
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


def refine_lemma_not_found_table():
    with db_cursor() as cur, \
            open('to_delete.json', 'w', encoding='utf-8') as to_delete:
        cur.execute('select * from dict.lemma_not_found_in_tezaurs')
        to_remove = []
        to_delete_arr = []
        for row in ProgIter(cur.fetchall(), verbose=2):
            word = Word(**row)
            if word.lemma in to_remove:
                to_delete_arr.append(row['number_line'])
                continue
            refined_lemma = Lemmatizer.normalize_lemma(word)
            if is_word_in_db(refined_lemma):
                to_remove.append(word.lemma)
                to_delete_arr.append(row['number_line'])
        json.dump(to_delete_arr, to_delete)


def delete_from_lemma_not_found_table():
    with db_cursor() as cur:
        with open('to_delete.json', 'r') as to_delete:
            lines = json.load(to_delete)
            cur.execute('delete from dict.lemma_not_found_in_tezaurs where number_line = any( %s )', (lines,))


def remove_the_ending_of_the_word(word: Word):
    """ Some words are not being lemmatized correctly  """


# TODO
#  1. check whether there is index by words in entries table
#  1. get the text
#  2. move it through LVTagger or sth
#  3. check if words are in DB
#  4. ML classification


if __name__ == '__main__':
    # write_non_existing_lemmas_to_csv()
    # refine_lemma_not_found_table()
    delete_from_lemma_not_found_table()

    # with db_cursor() as cur:
    #     cur.execute('select * from dict.')
