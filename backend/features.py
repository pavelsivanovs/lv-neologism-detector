import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from langdetect import detect_langs
from progiter import ProgIter

from utils.db import get_similar_words, db_cursor


@dataclass
class Features:
    # formal features
    length: int
    only_alpha: bool
    only_lv_characters: bool
    # relative_frequency: int
    # absolute_frequency: int

    # TODO REFINE THESE FEATURES
    # morphological features
    latvian_word: float
    latvian_sentence: float

    typo: float
    levenshtein_distance_to_closest_word: int
    similarity_to_closest_word: int

    # Can review compound splitting in the future
    # compound: float
    # number_of_roots: float
    # not_compound_but_just_lacks_space: float
    # root: float
    # prefixes: float
    # suffixes: float
    # ending: float

    POS_LABELS = ["a", "c", "i", "m", "n", "p", "q", "r", "s", "v", "x", "y"]
    pos: str
    is_ner: bool

    # can also add features describing structure of syllables and how those are suited for Latvian

    # thematic features
    # sentence_topic: Topics
    # paragraph_topic: Topics

    def __init__(self, word: dict):
        self.length = len(word['form'])
        self.only_alpha = True if re.match('^\w+$', word['form'].lower(), re.UNICODE) else False
        self.only_lv_characters = True if re.match('^[abcdefghijklmnoprstuvzāčēģīķļņšūž]+$',
                                                   word['form'].lower()) else False

        self._set_lang_features(word)

        # TODO RETHINK THIS APPROACH MAYBE?
        closest_word = get_similar_words(word['lemma'], limit=1)[0]
        self.typo = 1 - SequenceMatcher(None, word['lemma'], closest_word['word']).ratio()
        self.levenshtein_distance_to_closest_word = closest_word['distance']
        self.similarity_to_closest_word = closest_word['similarity']

        self.pos = word['tagset'][0]
        self.is_ner = True if word['ner'] else False

    def _get_syllables(self):
        pass

    def _set_lang_features(self, word: dict):
        def get_lv_lang_prob(text):
            try:
                lv = list(filter(lambda l: l.lang == 'lv', detect_langs(text)))
                return lv[0].prob if lv else 0
            except:
                return 0

        self.latvian_word = get_lv_lang_prob(word['form'])
        self.latvian_sentence = get_lv_lang_prob(word['sentence'])

    def _get_pos_onehot(self):
        return [1 if self.pos == label else 0 for label in self.POS_LABELS]

    def get_data_for_tensor(self):
        return [
            self.length,
            self.only_alpha,
            self.only_lv_characters,
            self.latvian_word,
            self.latvian_sentence,
            self.typo,
            self.levenshtein_distance_to_closest_word,
            self.similarity_to_closest_word,
            *self._get_pos_onehot(),
            self.is_ner
        ]


if __name__ == '__main__':
    with db_cursor() as cur, open('dairies_result_onehot.csv', 'w') as features_file:
        csv_writer = csv.writer(features_file)
        cur.execute('select * from dict.lemmas_not_found_in_dairies where is_neologism is not null')
        words = cur.fetchall()
        for word in ProgIter(words):
            csv_writer.writerow(Features(word).get_data_for_tensor())
