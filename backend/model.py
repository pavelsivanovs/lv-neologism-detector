from tagger import Lemmatizer


def is_word_in_db(word: str) -> bool:
    """ Checks whether word is present among dictionary entries. """
    pass


def is_neologism(word: str) -> float:
    """ Returns probability of a word being a real neologism, """
    pass


def model():
    """  """
    lemmatizer = Lemmatizer()
    test_texts = ['Ejam ar mani vērot saullēktu! Vakar bija mākoņaina debess, bet šodien nekas mums netraucēs.',
                  'šis ir testa teksts atkārtotai tagošanas rīka izmantošanai']
    for text in test_texts:
        print(f'Original text: {text}')
        print(f'Lemmatized text: {lemmatizer.lemmatize(text)}')



# TODO
#  1. check whether there is index by words in entries table
#  1. get the text
#  2. move it through LVTagger or sth
#  3. check if words are in DB
#  4. ML classification


if __name__ == '__main__':
    model()
