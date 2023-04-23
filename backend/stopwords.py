import requests


def get_stopwords() -> list[str]:
    req = requests.get('https://raw.githubusercontent.com/stopwords-iso/stopwords-lv/master/stopwords-lv.json',
                       stream=False)
    if req.text is None:
        raise Exception('Stopword retrieval could not get a list of stopwords.')
    return req.json()
