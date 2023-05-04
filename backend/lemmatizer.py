import os.path
import re
from os import fpathconf
from typing import List

from pexpect import EOF, TIMEOUT, spawn

from commoncrawl import Word
from utils.db import db_cursor
from utils.logger import get_configured_logger

logger = get_configured_logger(__name__)


class Lemmatizer:
    """ Encapsulation class for a LVTagger Morphotagging method
    implemented in PeterisP/LVTagger (https://github.com/PeterisP/LVTagger). """

    def __init__(self):
        backend_dir_path = os.path.abspath(os.path.dirname(__file__))
        morphotagger_script_path = os.path.join(backend_dir_path, 'tagger', 'morphotagger.sh')
        command = ['bash', morphotagger_script_path, '-vert']
        self._lemmatizer = spawn(' '.join(command), timeout=None, encoding='UTF-8')

        self._cpl = self._lemmatizer.compile_pattern_list('\r\n.*\r\n\r\n')
        self._byte_limit_per_line = fpathconf(0, 'PC_MAX_CANON')
        self._lemmatizer.expect(r'done \[.* sec\]\.\r\n')

        logger.info(f'Number of bytes that can be received by line is {self._byte_limit_per_line}')

    def __del__(self):
        self._lemmatizer.sendeof()

    @staticmethod
    def normalize_lemma(self, word: Word) -> Word:
        """ Sometimes dictionary has no entry for the lemma and these are usually the cases,
        when can we normalize lemma even more to get the respective entry in the dictionary. """

        # '-šana' nouns transforming to verbs if such are not in dict already
        if word.tagset[0] == 'n' and re.match('šana$', word.lemma):
            with db_cursor() as cur:
                cur.execute(  # there are some nouns ending in '-šana' in dict before, so we are checking for that
                    'select * from dict.lemma_not_found_in_tezaurs where supposed_lemma = %s limit 1',
                    (word.lemma,))
                if not cur.fetchone():
                    word.lemma = re.sub(r'šana$', 't', word.lemma)

        # if it is a verb and is negated, remove negation
        if word.tagset[0] == 'v' and word.tagset[9] == 'y' and word.lemma[:2] == 'ne':
            word.lemma = word.lemma[2:]

        # set lemma to be adjective instead of an adverb if it is possible
        if word.tagset[0] == 'r':
            word.lemma = re.sub(r'(āk|u|i|ām)$', 's', word.lemma)

        return word

    def lemmatize(self, text: str) -> List[str]:
        if text == '':
            logger.error('Empty string has been supplied to lemmatizer. Throwing exception.')
            raise Exception('Unintentional termination of the lemmatizer process by passing an empty string.')

        try:
            self._lemmatizer.sendline(text)
            self._lemmatizer.expect_list(self._cpl, timeout=5)
            output = self._lemmatizer.after
            return list(filter(None, map(lambda line: Word.get_word_from_line(line), output.split('\r\n'))))
        except EOF or TIMEOUT as exception:
            print(exception)
            raise exception


if __name__ == '__main__':
    lem = Lemmatizer()
    print(lem.lemmatize('Šis ir testa ievada teikums valodas apstrādes rīkam.'))
    # time.sleep(31)
    # print(lem.lemmatize('es uzdāvināju draudzenei Lego ziedu pušķa konstruktoru'))
    # # print(lem._lemmatizer)
