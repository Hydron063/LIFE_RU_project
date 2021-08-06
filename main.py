# -*- coding: utf-8 -*-
# 0.82 1.07
# {'learning_rate': 0.2, 'loss': 'lad', 'max_depth': 6, 'max_features': 'sqrt', 'n_estimators': 90, 'subsample': 1}
# K = 500
# 35
# ['Просмотры', 'section', 'countSymbols', 'year', 'month', 'day', 'time', 'ктг_Происшествия', 'тег_' ...]
# 31559 признаков

import os
import pandas as pd
import category_encoders as ce
import pymorphy2 as pymorphy2
import requests
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
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

# Пропуск предобработки
escape = True
if not escape:
    df['publicationDate'] = pd.to_datetime(df['publicationDate'])
    df['date'] = pd.to_datetime(df['publicationDate'].dt.date)
    print(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['time'] = (df['publicationDate'] - df['date']).dt.total_seconds()
    print(df['time'])

    # categories = set(y for interm in [x.split(';') for x in df['categories'].values.tolist()] for y in interm)
    for categorie in categories:
        df['ктг_' + categorie] = df['categories'].apply(lambda x: int(categorie in x))
    categories = ['ктг_' + categorie for categorie in categories]

    tags = list(set(y for interm in [x.split(';') for x in df['tags'].values.tolist()] for y in interm))
    tags = ['тег_' + tag for tag in tags]
    temp = list(zip(*df['tags'].map(lambda x: [int(tag in x) for tag in tags])))
    temp = [pd.Series(x) for x in temp]
    temp = pd.concat(temp, axis=1, ignore_index=True)
    temp.set_axis(tags, axis=1, inplace=True)
    print(temp.head(10))
    print(temp.shape)
    print(df.shape)
    df = pd.concat([df, temp], axis=1)

    encoder = ce.TargetEncoder()
    df['section'] = encoder.fit_transform(df['section'], df['Просмотры'])
    print(encoder.get_feature_names())

    url_stopwords_ru = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"
    stopwords_ru = get_text(url_stopwords_ru).splitlines()
    expr = r'(?u)\b\w{4,}\b'

    # Обработка по частям из-за проблем с производительностью
    # df1 = df['title'][:10000]
    # df1 = df1.map(lambda x: ' '.join(tokenize_n_lemmatize(x, stopwords=stopwords_ru, regexp=expr)))
    # print(df1)
    # print(df1.shape)
    # df1.to_csv('df.csv', index=False)
    # exit()

    # df_res = df['title'][10000:20000]
    # df_res = df_res.map(lambda x: ' '.join(tokenize_n_lemmatize(x, stopwords=stopwords_ru, regexp=expr)))
    # df1 = pd.read_csv('df.csv').squeeze()
    # print(df1.shape, df_res.shape)
    # df1 = df1.append(df_res)
    # df1.to_csv('df.csv', index=False)
    # print(type(df_res), type(df1))
    # print(df_res.shape, df1.shape)
    # exit()

    # df_res = df['title'][20000:30000]
    # df_res = df_res.map(lambda x: ' '.join(tokenize_n_lemmatize(x, stopwords=stopwords_ru, regexp=expr)))
    # df1 = pd.read_csv('df.csv').squeeze()
    # print(df1.shape, df_res.shape)
    # df1 = df1.append(df_res)
    # df1.to_csv('df.csv', index=False)
    # print(type(df_res), type(df1))
    # print(df_res.shape, df1.shape)
    # exit()

    # df_res = df['title'][30000:]
    # df_res = df_res.map(lambda x: ' '.join(tokenize_n_lemmatize(x, stopwords=stopwords_ru, regexp=expr)))
    # df1 = pd.read_csv('df.csv').squeeze()
    # print(df1.shape, df_res)
    # df1 = df1.append(df_res)
    # df1.to_csv('df.csv', index=False)
    # print(type(df_res), type(df1))
    # print(df_res.shape, df1.shape)
    # exit()

    df1 = pd.read_csv('df.csv').squeeze()
    print(df1.shape)

    # df2['title'] = df2['title'].apply(lambda x: ' '.join(tokenize_n_lemmatize(x, stopwords=stopwords_ru, regexp=expr)))
    # df['title'] = df['title'].apply(lambda x: ' '.join(tokenize_n_lemmatize(x, stopwords=stopwords_ru, regexp=expr)))
    # df.to_csv('df.csv')

    v = TfidfVectorizer(token_pattern=expr)
    # df1 = v.fit_transform(df['title'])
    df1 = v.fit_transform(df1)
    print(type(df1))
    print((v.get_feature_names()))
    print(df1)
    print(df1.shape)

    with open('section_save.txt', 'wb') as fp:
        pickle.dump(encoder, fp)
        pickle.dump(tags, fp)
        pickle.dump(v, fp)

    df = df.drop(['publicationDate', 'date', 'title', 'url', 'author', 'tags', 'categories', 'views'], axis=1)
    print('df')
    print(df.shape)
    print(df.head(5))
    print(df.columns.to_series().groupby(df.dtypes).groups)
    matr = sparse.hstack((sparse.coo_matrix(df.values), df1))
    print(matr.shape, df1.shape)
    sparse.save_npz('interm.npz', matr)
    exit()
else:
    matr = sparse.load_npz('interm.npz')
    print(matr.shape)
y = matr.toarray()[:, :1].ravel()
X = matr.toarray()[:, 1:]
scaler = StandardScaler()
with open('scaler.pkl', 'wb') as fp:
    pickle.dump(scaler, fp)
X = scaler.fit_transform(VarianceThreshold().fit_transform(X))

X = SelectKBest(f_regression, k=2000).fit_transform(X, y)

# Градиентный бустинг
parameters = {'learning_rate': [0.05 + 0.05 * x for x in range(8)], 'n_estimators': [40, 50, 60, 75, 90],
              'max_depth': [5, 6, 7, 8], 'max_features': ['sqrt'], 'subsample': [1],
              'loss': ['lad']}
regr = GridSearchCV(estimator=GBR(min_samples_split=0.0005, subsample=1, random_state=42),
                    param_grid=parameters, scoring=make_scorer(mape, greater_is_better=False), cv=5, n_jobs=-1)

regr.fit(X, y)
estimator = regr.best_estimator_
print(y[:20])
print(estimator.predict(X)[:20])
print(regr.best_params_)
print(regr.best_score_)
print(mape(y[:20], estimator.predict(X[:20])))
print(mape(y, estimator.predict(X)))
pickle.dump(estimator, open('model.sav', 'wb'))

# # Нейронка
# print(X.shape)
# def baseline_model():
#     model = Sequential()
#     model.add(Dense(X.shape[1], input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
#     model.add(Dense(X.shape[1]//2, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(X.shape[1]//4, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
#     return model
#
# estimator = KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=40000)
# kfold = KFold(n_splits=5)
# res = cross_val_score(estimator, X, y, cv=kfold)
# print(res, -res.mean())
# estimator.fit(X, y, epochs=100, batch_size=40000)
# estimator.model.save('neuron_model')
