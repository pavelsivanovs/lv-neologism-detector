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
            if row[1] in approved or row[1] in declined:
                continue

            context = get_text_sentence(cor.get_sentence(row[0])).replace(row[1], f'[bold cyan]{row[1]}[/]')

            console.rule(style=Style(color='bright_black'), title=f'Word {idx}/{cur.rowcount}')
            console.print(f'Is this a word or non-word? [underline bold cyan]{row[1]}')
            console.print(f'Supposed lemma: [bold]{row[2]}')
            console.print(f'Context:')
            console.print(f'[italic]{context}')
            console.print()

            table = Table(title='Similar words')
            table.add_column('Word', style='green')
            table.add_column('Distance')
            for word in get_similar_words(row[2]):
                table.add_row(word[1], str(word[2]))
            console.print(table)

            console.print('Is this word a word or not? [bright_black]\[y/n/q]')
            answer = get_key_pressed()
            console.print()
            if answer == 'y':
                approved.append(row[1])
                neologisms.write(','.join(map(lambda x: str(x), row)) + '\n')
            elif answer == 'n':
                declined.append(row[1])
            elif answer == 'q':
                console.rule()
                console.print(f'You have iterated over {idx} \[{cur.rowcount}] words')
                console.print('Good job!')
                neologisms.write(str(idx))
                break


if __name__ == '__main__':
    cor = CommonCrawl()
    verify_words()
