import time

import keyboard
from rich.console import Console
from rich.live import Live
from rich.style import Style
from rich.table import Table

from commoncrawl import CommonCrawl, get_text_sentence, get_similar_words
from utils.db import db_cursor


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
    approved = []
    declined = []

    with db_cursor() as cur, \
            open('neologisms.csv', mode='w', encoding='utf-8') as neologisms, \
            Live():
        neologisms.write('number_line,word,supposed_lemma,tagset\n')

        cur.execute('select * from dict.lemma_not_found_in_tezaurs')
        for idx, row in enumerate(cur.fetchall(), start=1):
            if row['word'] in approved or row['word'] in declined:
                continue

            context = get_text_sentence(cor.get_sentence(row['number_line']))\
                .replace(row['word'], f'[bold cyan]{row["word"]}[/]')

            console.rule(style=Style(color='bright_black'), title=f'Word {idx}/{cur.rowcount}')
            console.print(f'Is this a word or non-word? [underline bold cyan]{row["word"]}')
            console.print(f'Supposed lemma: [bold]{row["supposed_lemma"]}')
            console.print(f'Context:')
            console.print(f'[italic]{context}')
            console.print()

            table = Table(title='Similar words')
            table.add_column('Word', style='green')
            table.add_column('Distance')
            for word in get_similar_words(row["supposed_lemma"]):
                table.add_row(word['word'], str(word['distance']))
            console.print(table)

            console.print('Is this word a word or not? [bright_black]\[y/n/q]')
            answer = get_key_pressed()
            console.print()
            if answer == 'y':
                approved.append(row['word'])
                neologisms.write(','.join(map(lambda x: str(x), row)) + '\n')
            elif answer == 'n':
                declined.append(row['word'])
            elif answer == 'q':
                console.rule()
                console.print(f'You have iterated over {idx} \[{cur.rowcount}] words')
                console.print('Good job!')
                neologisms.write(str(idx))
                break


if __name__ == '__main__':
    cor = CommonCrawl()
    verify_words()
