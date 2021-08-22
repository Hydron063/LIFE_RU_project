# -*- coding: utf-8 -*-
# ['Просмотры',  'publicationDate', 'section', 'countSymbols', 'ктг_Происшествия' ...]
# 1) Добавить scale для y

import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import category_encoders as ce
import paramiko
import pymorphy2 as pymorphy2
import requests
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.metrics import mean_absolute_percentage_error as mape, make_scorer
from sklearn.linear_model import Ridge
from scipy import sparse
import pickle
import nltk
from nltk.corpus import stopwords
from nltk import regexp_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor as GBR, RandomForestRegressor as RFR
from sklearn.preprocessing import StandardScaler
from bert_serving.client import BertClient
from datetime import datetime, timedelta
from catboost import CatBoostRegressor, Pool, cv
import numpy as np
from scipy.stats import truncnorm
from openTSNE import TSNE


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


categories = ['Происшествия', 'Личные финансы', 'В мире', 'Технологии', 'Спорт', 'Наука', 'Культура', 'Регионы', 'Шоубиз',
     'Эксклюзивы', 'Авто', 'Армия', 'Политика', 'Общество', 'Здоровье', 'История', 'Россия', 'Экономика', 'Коронавирус',
     'Криминал', 'Поп-культура и Развлечения', 'Наука и Технологии', 'Интересное', 'новости', 'сша']

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
tabl1 = pd.concat(
    (pd.read_csv(os.path.join(path, f), encoding='utf_8', sep='	') for f in f_list if 'YM URL cor.' in f)).dropna()
tabl2 = pd.concat(
    (pd.read_csv(os.path.join(path, f), encoding='utf_8', sep='	') for f in f_list if 'YM URL amp' in f)).dropna()
tabl3 = pd.concat((pd.read_csv(os.path.join(path, f), encoding='cp1251') for f in f_list if 'DB.' in f)).dropna()

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
print(df['section'].unique())
print(df['publicationDate'])

last_date = df['publicationDate'].max()
last_date = (datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S').date()).strftime('%Y-%m-%d %H:%M:%S')
print(last_date, type(last_date))
df = df[df['publicationDate'] < last_date]
df = df.sort_values(by='publicationDate', ascending=False).head(3000)
df = df.reset_index(drop=True)
print(df)

escape = True
if not escape:
    # categories = set(y for interm in [x.split(';') for x in df['categories'].values.tolist()] for y in interm)
    for categorie in categories:
        df['ктг_' + categorie] = df['categories'].apply(lambda x: int(categorie in x))
    categories = ['ктг_' + categorie for categorie in categories]
    # print(list(df.columns.values))

    encoder = ce.TargetEncoder()
    df['section'] = encoder.fit_transform(df['section'], df['Просмотры'])
    print(encoder.get_feature_names())

    url_stopwords_ru = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"
    stopwords_ru = get_text(url_stopwords_ru).splitlines()
    expr = r'(?u)\b\w{4,}\b'
    df['title'] = df['title'].apply(lambda x: ' '.join(tokenize_n_lemmatize(x, stopwords=stopwords_ru, regexp=expr)))
    df.to_csv('df.csv')

    with open('encoder.sav', 'wb') as fp:
        pickle.dump(encoder, fp)

    client = BertClient()
    vectors = client.encode(df['title'].tolist())
    print(vectors.shape)

    df = df.drop(['title', 'url', 'author', 'tags', 'categories', 'views'], axis=1)
    print('df')
    print(df.shape)
    print(df.head(5))
    print(df.columns.to_series().groupby(df.dtypes).groups)
    vectors = pd.DataFrame(data=vectors, index=range(vectors.shape[0]), columns=range(vectors.shape[1]))
    df = pd.concat([df, vectors], axis=1)
    with open('final.sav', 'wb') as fp:
        pickle.dump(df, fp)
    print(df)
    exit()

with open('final.sav', 'rb') as fp:
    matr = pickle.load(fp)
print(np.array(matr).shape, type(matr))
print(matr)
y = matr['Просмотры']
interm = matr['publicationDate']
skip_reduct=False
n_comp = 100
X = matr.drop(['publicationDate', 'Просмотры'], axis=1).to_numpy()[:, :100]
if not skip_reduct:
    reduction = TSNE(n_components=3, random_state=42, verbose=True)
    print(X.shape)
    X = reduction.fit(X)
    print(X)
    print(type(X))
    with open('reduction.pkl', 'wb') as fp:
        pickle.dump(X, fp)
    print('OK')
else:
    with open('reduction.pkl', 'rb') as fp:
        reduction = pickle.load(fp)
    X = reduction.transform(X)

# # Демонстрация
# y_test = y.tolist()
# a, b = min(y_test), max(y_test)
# mean, std = np.mean(y_test), np.std(y_test)
# print(len(X), a, b, mean, std)
# y_comp = truncnorm.rvs((a-mean)/std, (b-mean)/std, loc=mean, scale=std, size=len(X))
# print('Результат для нормального распредения:')
# print(y_comp)
# print(mape(y_test, y_comp))
# print(np.mean(y_comp), np.std(y_comp))
# # 1000 50000 2000 200
# y_comp = truncnorm.rvs(-10, 10, 30, 2, size=1000000)
# print(min(y_comp), max(y_comp), np.mean(y_comp), np.std(y_comp))
# exit()

scaler = StandardScaler()
X = scaler.fit_transform(VarianceThreshold().fit_transform(X))
with open('scaler.pkl', 'wb') as fp:
    pickle.dump(scaler, fp)
X = pd.DataFrame(X)
X[n_comp] = interm
print(X)
print(type(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Catboost
cv_data = Pool(data=X_train, label=y_train, cat_features=[100])
params = {
    'iterations': 3294,
    'learning_rate': 0.001,
    'loss_function': 'MAPE',
    'eval_metric': 'MAPE'
}
scores = cv(cv_data, params=params, fold_count=5)
print(scores)
test_data = Pool(data=X_test, label=y_test, cat_features=[100])
# 2000 записей: 3294 0.001 = 0.65/0.7
model = CatBoostRegressor(iterations=3294, learning_rate=0.001, loss_function='MAPE', eval_metric='MAPE')
model.fit(cv_data)
model.save_model('model.cbm', format='cbm')

y_test = y_test.tolist()
a, b = min(y_test), max(y_test)
mean, std = (a+b)/2, np.std(y_test)
model = CatBoostRegressor()
model.load_model('model.cbm', format='cbm')
prediction = model.predict(test_data)
print('Результат модели:')
print(mape(y_test, prediction))

# # Нейронка
# def baseline_model():
#     model = Sequential()
#     model.add(Dense(256, input_dim=X.shape[1], kernel_initializer='normal', activation='relu',
#       kernel_regularizer=l1(0.00001)))
#     model.add(Dense(64, kernel_initializer='normal', activation='relu', kernel_regularizer=l1(0.00001)))
#     model.add(Dense(1, kernel_initializer='normal'))
#     model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
#     return model
#
# epochs = 100
# test_size = 30
# estimator = KerasRegressor(build_fn=baseline_model, epochs=epochs, batch_size=X.shape[0])
# kfold = KFold(n_splits=4)
# res = cross_val_score(estimator, X, y, cv=kfold)
# print(res, -res.mean())
# estimator.fit(X[test_size:], y[test_size:], epochs=epochs, batch_size=X.shape[0])
# estimator.model.save('neuron_model')
#
# # print(type(X))
# # estimator = keras.models.load_model('./neuron_model')
# prediction, y_true = estimator.predict(X[:test_size]), y[:test_size]
# print(y_true)
# print(prediction)
# # print(mape(y, prediction))
# print(mape(y_true, prediction))

# bert-serving-start -model_dir ./model/uncased_L-12_H-768_A-12/ -num_worker=1
# bert-serving-start -model_dir E:/PycharmProjets/Trucs_pythoniques/model/rubert_cased_L-12_H-768_A-12_v1/ -num_worker=1
# bert-serving-start -model_dir ./model/rubert_cased_L-12_H-768_A-12_v1/ -num_worker=1
