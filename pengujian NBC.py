# -*- coding: utf-8 -*-
"""
Created on Sun May  6 10:23:41 2018

@author: Latifah
"""
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle
import numpy as np
import pandas as pd

#Data Validasi
x = pd.read_csv("prepros(01).csv", encoding='utf-8')
y = pd.read_csv("prepros(01).csv", encoding='utf-8')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.1, random_state=5)

#%%
data_train = x_train
data_test = x_test

#%%
## inisialisasi data train ##
tweet_set = []
label_set = []

data1 = pd.DataFrame(data=data_train.tweet, dtype=str)
for tweet in data1.tweet:
    tweet_set.append(tweet)

data2 = pd.DataFrame(data=data_train.label)
for label in data2.label:
    label_set.append(label)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('classifier', MultinomialNB())])
X_train = np.asarray(tweet_set)


text_clf = text_clf.fit(X_train, np.asarray(label_set))


files = open('NBC_model.pickle', "wb")
pickle.dump(text_clf, files)
files.close()

#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import numpy as np
import pandas as pd

data_testing = pd.read_csv("testing_prepros(01).csv", encoding='utf-8')

    
openfiles = open('NBC_model.pickle', "rb")
model = pickle.load(openfiles)

predicted = model.predict(np.asarray(data_test.tweet))

akurasi = accuracy_score(data_test.label, predicted)*100
precision = precision_score(data_test.label, predicted)*100
recall = recall_score(data_test.label, predicted)*100

