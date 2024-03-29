import json
import shutil
import time

import keyboard
from rich.console import Console
from rich.live import Live
from rich.style import Style
from rich.table import Table

from corpus.commoncrawl import CommonCrawl
from corpus.word import get_text_sentence
from utils.db import db_cursor, get_similar_words
from utils.logger import get_configured_logger

logger = get_configured_logger(__name__)


def get_key_pressed():
    while True:
        time.sleep(0.1)
        if keyboard.is_pressed('q'):
            return 'q'
        elif keyboard.is_pressed('y'):
            return 'y'
        elif keyboard.is_pressed('n'):
            return 'n'


def verify_words():
    console = Console(width=70)
    cor = CommonCrawl()
    approved = []
    declined = []

    with db_cursor() as cur, \
            open('neologisms.json', mode='r+', encoding='utf-8') as neologisms_file, \
            Live():

        neologisms = json.load(neologisms_file)
        neologisms_file.truncate(0)
        neologisms_file.seek(0)
        if isinstance(offset := neologisms[-1], int):
            neologisms.pop()
        else:
            offset = 2000

        cur.execute("select * from dict.lemma_not_found_in_tezaurs where word ~ '[[:alpha:]]' offset %s", (offset,))
        for idx, row in enumerate(cur.fetchall(), start=offset):
            if row['word'] in approved or row['word'] in declined:
                continue

            context = get_text_sentence(cor.get_sentence(row['number_line'])) \
                .replace(row['word'], f'[bold cyan]{row["word"]}[/]')

            console.rule(
                style=Style(color='bright_black'),
                title=f'Word {idx}/{cur.rowcount} | Identified neologisms: {len(neologisms)}')
            console.print(f'Is this a new word or not? [underline bold]{row["word"]}')
            console.print(f'Supposed lemma: [underline bold cyan]{row["lemma"]}')
            console.print(f'Tagset: {row["tagset"]}')
            console.print(f'Context:')
            console.print(f'[italic]{context}')
            console.print()
            table = Table(title='Similar words')
            table.add_column('Word', style='green')
            table.add_column('Distance')
            table.add_column('Similarity')
            try:
                for word in get_similar_words(row["lemma"]):
                    table.add_row(word['word'], str(word['distance']), f'{word["similarity"]:.2%}')
            except Exception:
                logger.error(f'Exception happened during search of similar words of {row["lemma"]}', Exception)
                break
            console.print(table)

            console.print('Is this a new word or not? [bright_black]\[y/n/q]')
            answer = get_key_pressed()
            console.print()
            if answer == 'y':
                approved.append(row['word'])
                neologisms.append(row)
            elif answer == 'n':
                declined.append(row['word'])
            elif answer == 'q':
                break

        console.rule()
        console.print(f'You have iterated over {idx} / {cur.rowcount} words')
        console.print('Good job!')
        json.dump(neologisms + [idx], neologisms_file, ensure_ascii=False)
    shutil.copy('neologisms.json', 'neologisms.json.copy.auto')


def verify_words_in_dairies():
    console = Console(width=70, highlight=False)

    with db_cursor() as cur, Live():
        cur.execute('select * from dict.lemmas_not_found_in_dairies where is_neologism is null order by id')
        lemmas = cur.fetchall()
        row_count = cur.rowcount
        iterated_count = cur.execute('''
            select count(*) from dict.lemmas_not_found_in_dairies 
            where is_neologism is not null
            ''').fetchone()['count']
        neologism_count = cur.execute('''
            select count(*) from dict.lemmas_not_found_in_dairies 
            where is_neologism=1
            ''').fetchone()['count']

        for lemma in lemmas:
            console.rule(style=Style(color='bright_black'),
                         title=f'Word {iterated_count}/{row_count} | Identified {neologism_count} neologisms')
            iterated_count += 1

            sentence = lemma['sentence'].replace(lemma['form'], f'[bold cyan]{lemma["form"]}[/]')

            console.print(f'Is this a new word or not? [underline bold]{lemma["form"]}')
            console.print(f'Supposed lemma: [underline bold cyan]{lemma["lemma"]}')
            console.print(f'Tagset: {lemma["tagset"]}')
            console.print(f'NER: {lemma["ner"]}')
            console.print(f'Context: [italic]{sentence}')
            console.print()

            table = Table(title='Similar words')
            table.add_column('Word', style='green')
            table.add_column('Distance')
            table.add_column('Similarity')
            try:
                for word in get_similar_words(lemma['lemma']):
                    table.add_row(word['word'], str(word['distance']), f'{word["similarity"]:.2%}')
            except Exception:
                logger.error(f'Exception happened during search of similar words of {lemma["lemma"]}', Exception)
                break
            console.print(table)

            console.print('Is this a new word or not? [bright_black]\[y/n/q]')
            answer = get_key_pressed()
            console.print()
            if answer == 'y':
                cur.execute('''
                    update dict.lemmas_not_found_in_dairies 
                    set is_neologism = 1
                    where id = %s
                ''', (lemma['id'],))
                neologism_count += 1
            elif answer == 'n':
                cur.execute('''
                    update dict.lemmas_not_found_in_dairies 
                    set is_neologism = 0
                    where id = %s
                ''', (lemma['id'],))
            elif answer == 'q':
                break

        console.rule()
        console.print(f'You have iterated over {iterated_count} / {row_count} words')
        console.print('Good job!')


if __name__ == '__main__':
    # verify_words()
    verify_words_in_dairies()
