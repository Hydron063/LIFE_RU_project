# -*- coding: utf-8 -*-
# Кратные записи при необходимости указать несколько авторов в tabl1

# Меняем на градиентный спуск
# Запускаем большой датасет (на сервере)
# Код - на гит
# Посмотреть DeepPavlov

import os

import pandas as pd
import category_encoders as ce
import pymorphy2 as pymorphy2
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.metrics import mean_absolute_percentage_error as mape, make_scorer
from scipy import sparse
import nltk
from nltk.corpus import stopwords
from nltk import regexp_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

nltk.download('punkt')


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


## Создание csv из xlsx
# read_file = pd.read_excel('LIFE_ru.xlsx', engine='openpyxl')
# read_file.to_csv('LIFE_ru1.csv', index=None, header=True)
# a_file = open("LIFE_ru1.csv", "r", encoding='utf_8')
# lines = a_file.readlines()
# a_file.close()
# new_file = open("LIFE_ru2.csv", "w", encoding='utf_8')
# for line in lines[5:]:
#     new_file.write(line)
# new_file.close()

# f = open('res.txt', 'w', encoding='cp1251')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

# tabl = pd.read_csv('amp-may-GA2.csv', encoding='mbcs').dropna()
tabl = pd.read_csv('LIFE_ru2.csv', encoding='utf_8').dropna()
tabl1 = pd.read_csv('db-may-tags.csv', encoding='cp1251').dropna()
print(tabl.shape, tabl1.shape)
print(tabl.shape[0], tabl['URL материала'].nunique())
print(tabl1.shape[0], tabl1['url'].nunique())
# tabl1 = tabl1.groupby(['url']).sum().reset_index(name='counts')
# print(tabl1[tabl1['counts']>1].sort_values('counts'))
tabl = tabl.drop(set(tabl.columns) - {'Визиты', 'URL материала'}, axis=1)
tabl = tabl.groupby(['URL материала']).sum().reset_index()
print(tabl)
tabl1.drop_duplicates('url', keep='first', inplace=True)
print(tabl1.shape[0], tabl1['url'].nunique())
print(tabl1['section'].head().to_string())
print(tabl.head())

df = tabl.merge(tabl1, left_on='URL материала', right_on='url', left_index=False).drop(['URL материала', 'views'],
                                                                                       axis=1)
print(df.head(20))
print(df.columns)
print(df.shape)
print(df['categories'].unique())
print(df['author'].nunique())
print(df['publicationDate'])
df['publicationDate'] = pd.to_datetime(df['publicationDate'])
df['date'] = pd.to_datetime(df['publicationDate'].dt.date)
print(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['time'] = (df['publicationDate'] - df['date']).dt.total_seconds()
print(df['time'])

categories = set(y for interm in [x.split(';') for x in df['categories'].values.tolist()] for y in interm)
tags = set(y for interm in [x.split(';') for x in df['tags'].values.tolist()] for y in interm)
for categorie in categories:
    df['ктг_' + categorie] = df['categories'].apply(lambda x: int(categorie in x))
categories = ['ктг_' + categorie for categorie in categories]
for tag in tags:
    df['тег_' + tag] = df['tags'].apply(lambda x: int(tag in x))
tags = ['тег_' + tag for tag in tags]
print(df[list(tags)])

# encoder = ce.OneHotEncoder(cols=['section'])
# encoder.fit(df, df['Визиты'])
# print(encoder.transform(df))
escape = True
if not escape:
    encoder = ce.TargetEncoder()
    df['section'] = encoder.fit_transform(df['section'], df['Визиты'])
    print(encoder.get_feature_names())

    url_stopwords_ru = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"
    stopwords_ru = get_text(url_stopwords_ru).splitlines()
    print(stopwords_ru)
    expr = r'(?u)\b\w{4,}\b'
    df['title'] = df["title"].apply(lambda x: ' '.join(tokenize_n_lemmatize(x, stopwords=stopwords_ru, regexp=expr)))

    v = TfidfVectorizer(token_pattern=expr)
    df1 = v.fit_transform(df['title'])
    print(type(df1))
    print((v.get_feature_names()))
    print(df1)

    df = df.drop(['publicationDate', 'date', 'title', 'url', 'author', 'tags', 'categories'], axis=1)
    print(df.columns.to_series().groupby(df.dtypes).groups)
    matr = sparse.hstack((sparse.coo_matrix(df.values), df1))
    print(matr.shape, df1.shape)
    sparse.save_npz('interm.npz', matr)
else:
    matr = sparse.load_npz('interm.npz')
    print(matr.shape)
y = matr.toarray()[:, :1].ravel()
X = matr.toarray()[:, 1:]
scaler = StandardScaler()
X = scaler.fit_transform(VarianceThreshold().fit_transform(X))
X = SelectKBest(f_regression, k=50).fit_transform(X, y)

parameters = {'kernel': ('linear', 'poly', 'rbf'), 'C': [2 ** x for x in range(15, 19, 2)]}
regr = GridSearchCV(estimator=SVR(), param_grid=parameters, scoring=make_scorer(mape), cv=5)
regr.fit(X, y)
print(regr.best_params_)
print(regr.best_score_)
estimator = regr.best_estimator_
print(mape(y, estimator.predict(X)))
