import csv
import json
import re
import time

import requests
import torch
from progiter import ProgIter

from corpus.commoncrawl import CommonCrawl
from corpus.word import Word
from features import Features
from lemmatizer import Lemmatizer
from training import NeologismClassificator
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


def extract_neologisms(text: str):
    req = requests.post('http://localhost:9500/api/nlp', json={
        'steps': ['tokenizer', 'morpho', 'parser', 'ner'],  # maybe add another step
        'data': {
            'text': text
        }
    })

    if not (data := req.json().get('data')):
        return

    model = NeologismClassificator(input_size=21)
    model.load_state_dict(torch.load('./model_with_random_sampler.pt'))
    model.eval()

    with torch.no_grad():
        output = []
        for sentence in data['sentences']:
            sentence_cat = ''
            sentence_potential_neologisms = []
            for token in sentence['tokens']:
                sentence_cat += token['form'] + ' '
                if token['tag'][0] != 'z' and token['tag'] not in ('xo', 'xn') \
                        and (not re.search(r'Leksēmas_nr=\d+\|', token['features'])
                             and not is_word_in_db(token['lemma'])):
                    sentence_potential_neologisms.append(token)

            filtered = list(filter(lambda ner: token['form'] in ner['text'], sentence['ner']))
            ner_label = filtered[0]['label'] if filtered else None
            features = [Features(word={**t, 'sentence': sentence_cat, 'ner': ner_label}).get_data_for_tensor() for t in
                        sentence_potential_neologisms]
            predictions = model(torch.tensor(features, dtype=torch.float32))
            output.extend(zip(sentence_potential_neologisms, predictions))
        output.sort(key=lambda x: x[1], reverse=True)
        return output


if __name__ == '__main__':
    test_sentence = 'Mēs vairs neesam koviddisidenti. Tik un tā, mana māte vnk copy visas manas atbildes šajā eksprestestā.'
    results = extract_neologisms(test_sentence)
    print(f'Input text: {test_sentence}')
    print('Possible neologisms that should be considered for being added to the vocabulary')
    for token, prediction in results:
        print(f"Word: {token['form']:15} | Lemma: {token['lemma']:15} | Confidence: {prediction.item():.2%}")
