import os.path
import re
from os import fpathconf
from typing import List

from pexpect import EOF, TIMEOUT, spawn

from commoncrawl import Word
from utils.db import db_cursor, is_word_in_db
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
    def normalize_lemma(word: Word) -> str:
        """ Sometimes dictionary has no entry for the lemma and these are usually the cases,
        when can we normalize lemma even more to get the respective entry in the dictionary. """
        lemma = word.lemma

        # '-šana' nouns transforming to verbs if such are not in dict already
        if word.tagset[0] == 'n' and re.search(r'šana$', lemma):
            if not is_word_in_db(lemma):
                lemma = re.sub(r'šana$', 't', lemma)

        # if it is a negated verb, remove negation
        if word.tagset[0] == 'v' and word.tagset[9] == 'y' and lemma[:2] == 'ne':
            lemma = lemma[2:]

        # set lemma to be adjective instead of an adverb if it is possible
        if word.tagset[0] == 'r':
            lemma = re.sub(r'(āk|u|i|ām)$', 's', lemma)

        return lemma

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
    # lem = Lemmatizer()
    # print(lem.lemmatize('Šis ir testa ievada teikums valodas apstrādes rīkam.'))
    # time.sleep(31)
    # print(lem.lemmatize('es uzdāvināju draudzenei Lego ziedu pušķa konstruktoru'))
    # # print(lem._lemmatizer)

    with db_cursor() as cur:
        cur.execute('''
            select * 
            from dict.lemma_not_found_in_tezaurs
            where lemma like '%šana'
        ''')
        dic = cur.fetchone()
        print(dic)
        word = Word(**dic)
        print(word)
        res = Lemmatizer.normalize_lemma(word)
        print(res)
