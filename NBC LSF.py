# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 17:32:37 2018

@author: Latifah
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import LSF

data_train = pd.read_csv("prepros(01).csv", encoding='utf-8')
data_testing = pd.read_csv("testing_prepros(01).csv", encoding='utf-8')

#%%
## inisialisasi data train ##
train_set = []

for i in range(len(data_train)):
    data1 = data_train.tweet[i]
    data2 = data_train.label[i]
    train_tup = (data1, data2)
    train_set.append(train_tup)

tweet_train_set = [tweet for tweet,label in train_set]
label_train_set = [label for tweet,label in train_set]


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
test_set = []

for i in range(len(data_testing)):
    data3 = data_testing.tweet[i]
    data4 = data_testing.label[i]
    test_tup = (data3, data4)
    test_set.append(test_tup)

tweet_test_set = [tweet for tweet,label in test_set]
label_test_set = [label for tweet,label in test_set]


## inisialisasi LSF data test ##
test_lsf = []

lsf_test = LSF.LSF(data_testing)

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

final_train = final_train[:,:2951]
#%%
## BoW data test dari model ##
test_counts = count_vect.transform((tweet_test_set))
## tfidf data test
test_tfidf = tfidf_transformer.transform(test_counts)

test = test_tfidf.toarray()
#%%
final_test = np.hstack([test, test_lsf])

final_test = final_test[:,:2951]

#%%
model = MultinomialNB().fit(np.asarray(final_train), np.asarray(label_train_set))
predicted = model.predict(np.asarray(final_test))

akurasi = accuracy_score(data_testing.label, predicted)
precision = precision_score(data_testing.label, predicted)
recall = recall_score(data_testing.label, predicted)