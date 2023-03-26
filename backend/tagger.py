import os.path
import time
from os import fpathconf
from typing import List

from pexpect import EOF, TIMEOUT, spawn


class Lemmatizer:
    """ Encapsulation class for a LVTagger Morphotagging method
    implemented in PeterisP/LVTagger (https://github.com/PeterisP/LVTagger). """

    def __init__(self):
        morphotagger_script_path = os.path.abspath(os.path.join(os.path.curdir, 'tagger', 'morphotagger.sh'))
        command = ['bash', morphotagger_script_path, '-lemmatized-text']
        self._lemmatizer = spawn(' '.join(command), timeout=None, encoding='UTF-8')

        self._cpl = self._lemmatizer.compile_pattern_list('\r\n.*\r\n\r\n')
        self._byte_limit_per_line = fpathconf(0, 'PC_MAX_CANON')
        self._lemmatizer.expect(r'done \[.* sec\]\.\r\n')

        print(f'Number of bytes that can be received by line is {self._byte_limit_per_line}')
        # TODO set logging

    def __del__(self):
        self._lemmatizer.sendeof()

    def lemmatize(self, text: str) -> List[str]:
        if text == '':
            raise Exception('Unintentional termination of the lemmatizer process by passing an empty string.')

        try:
            self._lemmatizer.sendline(text)
            self._lemmatizer.expect_list(self._cpl, timeout=5)
            output = self._lemmatizer.after
            return output.split()
        except EOF or TIMEOUT as exception:
            print(exception)
            raise exception


if __name__ == '__main__':
    lem = Lemmatizer()
    print(lem.lemmatize('Šis ir testa ievada teikums valodas apstrādes rīkam.'))
    time.sleep(31)
    print(lem.lemmatize('es uzdāvināju draudzenei Lego ziedu pušķa konstruktoru'))
    # print(lem._lemmatizer)
