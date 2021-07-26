# -*- coding: utf-8 -*-
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
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.preprocessing import StandardScaler


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


nltk.download('punkt')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

# Исправляем ошибки в файлах
path = './Выгрузки'
f_list = os.listdir(path)
for f in f_list:
    if 'YM URL.' in f:
        file1 = open(os.path.join(path, f), 'r', encoding='utf_8')
        file2 = open(os.path.join(path, f[:-4] + ' cor.csv'), 'w', encoding='utf_8')
        for s in file1:
            if '#' in s:
                s = s.split('#')[0] + '	' + s.split()[-1] + '\n'
            file2.write(s)
    if 'YM URL amp.' in f:
        temp = pd.read_csv(os.path.join(path, f), encoding='utf_8', sep='	')
        if temp.columns.tolist()[0] == 'Просмотры':
            temp = temp[temp.columns.tolist()[1::-1]]
            temp.to_csv(os.path.join(path, f), encoding='utf_8', sep='	', index=False)

f_list = os.listdir(path)
print('0')
tabl1 = pd.concat(
    (pd.read_csv(os.path.join(path, f), encoding='utf_8', sep='	') for f in f_list if 'YM URL cor.' in f)).dropna()
print('1')
tabl2 = pd.concat(
    (pd.read_csv(os.path.join(path, f), encoding='utf_8', sep='	') for f in f_list if 'YM URL amp' in f)).dropna()
print('2')
tabl3 = pd.concat((pd.read_csv(os.path.join(path, f), encoding='cp1251') for f in f_list if 'DB.' in f)).dropna()
print('3')

print(tabl1.shape, tabl2.shape, tabl3.shape)
print(tabl1.head(10))
print(tabl1.shape[0], tabl1['URL материала'].nunique())
print(tabl2.head(10))
print(tabl2.shape[0], tabl2['URL trim amp'].nunique())
print(tabl3.shape[0], tabl3['url'].nunique())

tabl1 = tabl1.groupby(['URL материала']).sum().reset_index()
print(tabl1)
tabl2 = tabl2.groupby(['URL trim amp']).sum().reset_index()
print(tabl2)
tabl3.drop_duplicates('url', keep='first', inplace=True)
print(tabl3.shape[0], tabl3['url'].nunique())
print(tabl3['section'].head().to_string())
print(tabl3.head())

df = tabl1.merge(tabl2, left_on='URL материала', right_on='URL trim amp', left_index=False).drop(['URL trim amp'],
                                                                                                 axis=1)
df['Просмотры'] = df['Просмотры'] + df['Просмотры материалов']
df.drop(['Просмотры материалов'], axis=1, inplace=True)
print(df)
df = df.merge(tabl3, left_on='URL материала', right_on='url', left_index=False).drop(['URL материала'], axis=1)
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

temp = list(zip(*df['tags'].map(lambda x: [int(tag in x) for tag in tags])))
temp = [pd.Series(x) for x in temp]
tags = ['тег_' + tag for tag in tags]
temp = pd.concat(temp, axis=1, ignore_index=True)
temp.set_axis(tags, axis=1, inplace=True)
print(temp.head(10))
print(temp.shape)
print(df.shape)

df = pd.concat([df, temp], axis=1)
print(df.columns)
print(df[list(tags)])
# for i, c in enumerate(['тег_' + tag for tag in tags]):
#     df[c] = temp[i]
# for tag in tags:
#     df['тег_' + tag] = df['tags'].apply(lambda x: int(tag in x))
# tags = ['тег_' + tag for tag in tags]

escape = True
if not escape:
    encoder = ce.TargetEncoder()
    df['section'] = encoder.fit_transform(df['section'], df['Просмотры'])
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

# parameters = {'kernel': ('linear', 'poly', 'rbf'), 'C': [2 ** x for x in range(19, 20, 2)], 'degree': [3, 4, 5]}
# regr = GridSearchCV(estimator=SVR(),
#                     param_grid=parameters, scoring=make_scorer(mape, greater_is_better=False), cv=5)
parameters = {'learning_rate': [0.1 + 0.05 * x for x in range(1, 9)], 'n_estimators': [25, 50, 75, 100],
              'max_depth': [7, 9, 11], 'max_features': ['sqrt'], 'subsample': [0.8, 1],
              'loss': ['lad']}
regr = GridSearchCV(estimator=GBR(min_samples_split=0.0005, subsample=1, random_state=42),
                    param_grid=parameters, scoring=make_scorer(mape, greater_is_better=False), cv=5)
regr.fit(X, y)
estimator = regr.best_estimator_
print(y[:20])
print(estimator.predict(X)[:20])
print(regr.best_params_)
print(regr.best_score_)
print(mape(y[:20], estimator.predict(X[:20])))
print(mape(y, estimator.predict(X)))
