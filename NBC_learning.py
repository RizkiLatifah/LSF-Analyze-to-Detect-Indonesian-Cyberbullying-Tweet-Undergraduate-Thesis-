# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 00:13:16 2018

@author: Latifah
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle
import numpy as np
import pandas as pd


data_train = pd.read_csv("prepros(01).csv", encoding='utf-8')
#%%
## NBC learning Model ##

train_set = []

for i in range(len(data_train)):
    data1 = data_train.tweet[i]
    data2 = data_train.label[i]
    tup = (data1, data2)
    train_set.append(tup)

tweet_set = [tweet for tweet,label in train_set]
label_set = [label for tweet,label in train_set]


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('classifier', MultinomialNB())])
X_train = np.asarray(tweet_set)


text_clf = text_clf.fit(X_train, np.asarray(label_set))


files = open('NBC_model.pickle', "wb")
pickle.dump(text_clf, files)
files.close()


#%%
## NBC Testing ##

from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import numpy as np
import pandas as pd

data_testing = pd.read_csv("testing_prepros(01).csv", encoding='utf-8')

    
openfiles = open('NBC_model.pickle', "rb")
model = pickle.load(openfiles)

predicted = model.predict(np.asarray(data_testing.tweet))

akurasi = accuracy_score(data_testing.label, predicted)
precision = precision_score(data_testing.label, predicted)
recall = recall_score(data_testing.label, predicted)





