import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from langdetect import detect_langs

from corpus.commoncrawl import CommonCrawl
from corpus.word import Word
from utils.db import get_similar_words


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

    compound: float
    number_of_roots: float
    not_compound_but_just_lacks_space: float
    # measuring if listed morphemes are found in Latvian language
    root: float
    prefixes: float
    suffixes: float
    ending: float

    lexemes_in_word: float

    # can also add features describing structure of syllables and how those are suited for Latvian

    # thematic features
    # sentence_topic: Topics
    # paragraph_topic: Topics

    def __init__(self, word: Word):
        self.length = len(word.word)
        self.only_alpha = True if re.match('^\w+$', word.word.lower(), re.UNICODE) else False
        self.only_lv_characters = True if re.match('^[a-zāčēģīķļņšŗūž]+$', word.word.lower()) else False

        self._set_lang_features(word)

        # TODO RETHINK THIS APPROACH MAYBE?
        closest_word = get_similar_words(word.lemma, limit=1)[0]
        self.typo = 1 - SequenceMatcher(None, word.lemma, closest_word['word']).ratio()
        self.levenshtein_distance_to_closest_word = closest_word['distance']

    def _get_syllables(self):
        pass

    def _set_lang_features(self, word: Word):
        def get_lv_lang_prob(text):
            lv = list(filter(lambda l: l.lang == 'lv', detect_langs(text)))
            return lv[0].prob if lv else 0

        self.latvian_word = get_lv_lang_prob(word.word)
        self.latvian_sentence = get_lv_lang_prob(word.sentence)


if __name__ == '__main__':
    cc = CommonCrawl()
