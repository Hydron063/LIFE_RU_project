import os

import pymorphy2
from flask import Flask, jsonify, request
import keras
import pickle
import category_encoders as ce
import numpy as np
from datetime import datetime as dt
from bert_serving.client import BertClient
from catboost import CatBoostRegressor, Pool, cv
import pandas as pd
from nltk import regexp_tokenize, sent_tokenize
import requests


def get_text(url, encoding='utf-8', to_lower=True):
    url = str(url)
    if url.startswith('http'):
        r = requests.get(url)
        if not r.ok:
            r.raise_for_status()
        return r.text.lower() if to_lower else r.text
    elif os.path.exists(url):
        with open(url, encoding=encoding) as f:
            return f.read().lower() if to_lower else f.read()
    else:
        raise Exception('parameter [url] can be either URL or a filename')


def normalize_tokens(tokens):
    morph = pymorphy2.MorphAnalyzer()
    return [morph.parse(tok)[0].normal_form for tok in tokens]


def remove_stopwords(tokens, stopwords=None, min_length=4):
    if not stopwords:
        return tokens
    stopwords = set(stopwords)
    tokens = [tok
              for tok in tokens
              if tok not in stopwords and len(tok) >= min_length]
    return tokens


def tokenize_n_lemmatize(text, stopwords=None, normalize=True, regexp=r'(?u)\b\w{4,}\b'):
    words = [w for sent in sent_tokenize(text)
             for w in regexp_tokenize(sent, regexp)]
    if normalize:
        words = normalize_tokens(words)
    if stopwords:
        words = remove_stopwords(words, stopwords)
    return words

categories = ['Происшествия', 'Личные финансы', 'В мире', 'Технологии', 'Спорт', 'Наука', 'Культура', 'Регионы',
              'Шоубиз',
              'Эксклюзивы', 'Авто', 'Армия', 'Политика', 'Общество', 'Здоровье', 'История', 'Россия', 'Экономика',
              'Коронавирус',
              'Криминал', 'Поп-культура и Развлечения', 'Наука и Технологии', 'Интересное', 'новости', 'сша']
app = Flask(__name__)


@app.route('/start', methods=['POST'])
def calculate():
    data = request.form
    title = data.get('name')
    category = data.get('category')
    # tag = data.get('tag')
    section = data.get('chapter')
    n_symbols = data.get('n_symbols')
    with open('encoder.sav', 'rb') as fp:
        encoder = pickle.load(fp)

    x = pd.DataFrame()
    x['section'] = [section]
    x['section'] = encoder.transform(x['section'])
    x['countSymbols'] = [n_symbols]
    for ctg in categories:
        x['ктг_' + ctg] = int(category == ctg)

    url_stopwords_ru = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"
    stopwords_ru = get_text(url_stopwords_ru).splitlines()
    expr = r'(?u)\b\w{4,}\b'
    x['title'] = tokenize_n_lemmatize(title, stopwords=stopwords_ru, regexp=expr)

    client = BertClient()
    vector = client.encode(x['title'].tolist())
    for i, v in enumerate(vector):
        x[i] = v
    # x_list += [int(t in tag) for t in tags]

    with open('reduction.pkl', 'rb') as fp:
        reduction = pickle.load(fp)
    x = reduction.transform(x)

    with open('scaler.pkl', 'rb') as fp:
        scaler = pickle.load(fp)
    x = scaler.transform(x)

    x = pd.DataFrame(x)
    n_comp = 100
    x[n_comp] = dt.now().strftime('%Y-%m-%d %H:%M:%S')

    model = CatBoostRegressor()
    model.load_model('model.cbm', format='cbm')
    x = Pool(data=x, cat_features=[100])
    pred = int(model.predict(x))
    res = {'result': pred}
    return jsonify(res), 200


if __name__ == "__main__":
    app.run()
