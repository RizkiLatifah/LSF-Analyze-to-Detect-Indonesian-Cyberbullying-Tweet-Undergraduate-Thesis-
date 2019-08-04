# -*- coding: utf-8 -*-
"""
Created on Sat May  5 20:09:48 2018

@author: Latifah
"""
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import LSF

#Data Validasi
x = pd.read_csv("prepros(01).csv", encoding='utf-8')
y = pd.read_csv("prepros(01).csv", encoding='utf-8')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.4, random_state=5)

#%%

data_train = x_train
data_test = x_test

#%%
## inisialisasi data train ##
tweet_train_set = []
label_train_set = []

data1 = pd.DataFrame(data=data_train.tweet, dtype=str)
for tweet in data1.tweet:
    tweet_train_set.append(tweet)

data2 = pd.DataFrame(data=data_train.label)
for label in data2.label:
    label_train_set.append(label)
        
## inisialisasi LSF data train ##
train_lsf = []

lsf_train = LSF.LSF(data_train)

for i in range(len(lsf_train)):
    lsf1 = lsf_train[i]
    lsf2 = lsf_train[i]
    lsf_train_tup = (lsf1,lsf2)
    train_lsf.append(lsf_train_tup)

train_lsf = np.array(train_lsf)

#%%
## inisialisasi data test ##
tweet_test_set = []
label_test_set = []

data3 = pd.DataFrame(data=data_test.tweet, dtype=str)
for tweet in data3.tweet:
    tweet_test_set.append(tweet)

data4 = pd.DataFrame(data=data_test.label)
for label in data4.label:
    label_test_set.append(label)


## inisialisasi LSF data test ##
test_lsf = []

lsf_test = LSF.LSF(data_test)

for i in range(len(lsf_test)):
    lsf3 = lsf_test[i]
    lsf4 = lsf_test[i]
    lsf_test_tup = (lsf3,lsf4)
    test_lsf.append(lsf_test_tup)

test_lsf = np.array(test_lsf)


#%%
## BoW data train ##
count_vect = CountVectorizer().fit((tweet_train_set))
train_counts = count_vect.transform((tweet_train_set))

## tfidf data train ##
tfidf_transformer = TfidfTransformer().fit(train_counts)
train_tfidf = tfidf_transformer.transform(train_counts)

train = train_tfidf.toarray()
#%%
final_train = np.hstack([train, train_lsf])

final_train = final_train[:,:2784]
#%%
## BoW data test dari model ##
test_counts = count_vect.transform((tweet_test_set))
## tfidf data test
test_tfidf = tfidf_transformer.transform(test_counts)

test = test_tfidf.toarray()
#%%
final_test = np.hstack([test, test_lsf])

final_test = final_test[:,:2784]

#%%
model = MultinomialNB().fit(np.asarray(final_train), np.asarray(label_train_set))
predicted = model.predict(np.asarray(final_test))

akurasi = accuracy_score(data_test.label, predicted)*100
precision = precision_score(data_test.label, predicted)*100
recall = recall_score(data_test.label, predicted)*100