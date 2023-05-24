import json
import re

import requests
from progiter import ProgIter

from corpus.init_corpus import Corpus
from utils.db import is_word_in_db, db_cursor
from utils.logger import get_configured_logger

logger = get_configured_logger(__name__)


class PandemicDairies(Corpus):
    filename = 'LFK-Ak-166.txt'

    def __init__(self, path='../../corpus/pandemic_dairies'):
        super().__init__(path)


def write_non_existing_lemmas_to_csv():
    pandemic_diaries = PandemicDairies()
    lemmas_not_found = []
    bad_token_count = sentence_count = token_count = 0
    try:
        with pandemic_diaries.open_corpus() as corpus:
            idx = 0
            for line in ProgIter(corpus, verbose=2, initial=1):
                idx += 1
                if line == '':
                    logger.info(f'Empty line at {idx}')
                    continue
                if re.match('LFK Ak-166-\d+', line):
                    logger.info(f'Reading new dairy: {line}')
                    continue

                req = requests.post('http://localhost:9500/api/nlp', json={
                    'steps': ['tokenizer', 'morpho', 'parser', 'ner'],  # maybe add another step
                    'data': {
                        'text': line
                    }
                })

                if not (data := req.json().get('data')):
                    continue
                for sentence in data['sentences']:
                    sentence_count += 1
                    s = ''
                    t = []
                    for token in sentence['tokens']:
                        token_count += 1
                        s += token['form'] + ' '
                        if token['tag'][0] != 'z' and token['tag'] not in ('xo', 'xn') \
                                and (not re.search(r'Leksēmas_nr=\d+\|', token['features'])
                                     and not is_word_in_db(token['lemma'])):
                            t.append(token)
                            # logger.info(f'adding token {token}')
                            bad_token_count += 1
                    if t:
                        lemmas_not_found.append({
                            'line': idx,
                            'sentence': s,
                            'ner': sentence['ner'],
                            'tokens': t,
                        })
    finally:
        with open('lemma_not_found_in_pandemics.json', mode='w', encoding='utf-8') as output_file:
            json.dump(lemmas_not_found, output_file, ensure_ascii=False)
        logger.info(f'Out of {token_count} tokens from {sentence_count} sentences {bad_token_count} '
                    f'tokens did not have lemma in DB.')


def create_lemma_not_found_table_for_dairies():
    with db_cursor() as cur:
        cur.execute('''
            create table if not exists dict.lemmas_not_found_in_dairies (
                id serial primary key,
                form varchar not null,
                lemma varchar not null,
                tagset varchar not null,
                pos varchar not null,
                features varchar not null,
                ufeats varchar not null,
                upos varchar not null,
                deprel varchar,
                parent int,
                line int not null,
                sentence varchar not null,
                word_index int not null,
                ner varchar,
                is_neologism int
            );
        ''')


def json_to_db_lemmas_not_found():
    with open('../lemma_not_found_in_pandemics.json', mode='r') as lemmas_file, \
            db_cursor() as cur:
        lemmas = json.load(lemmas_file)
        for sentence in ProgIter(lemmas, verbose=2):
            for token in sentence['tokens']:
                filtered = list(filter(lambda ner: token['form'] in ner['text'], sentence['ner']))
                ner_label = filtered[0]['label'] if filtered else None
                cur.execute('''
                    insert into dict.lemmas_not_found_in_dairies 
                    (form, lemma, tagset, pos, features, ufeats, upos, deprel, parent, line, sentence, word_index, ner) 
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                ''', (token['form'], token['lemma'], token['tag'], token['pos'], token['features'], token['ufeats'],
                      token['upos'], token['deprel'], token['parent'], sentence['line'], sentence['sentence'],
                      token['index'], ner_label))


if __name__ == '__main__':
    json_to_db_lemmas_not_found()
