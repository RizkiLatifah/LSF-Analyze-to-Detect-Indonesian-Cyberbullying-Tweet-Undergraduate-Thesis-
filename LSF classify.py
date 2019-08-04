# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 23:33:53 2018

@author: Latifah
"""

import pandas as pd
import LSF

dataset = pd.read_csv("testing_prepros(01).csv", encoding='utf-8')
#%%
score = pd.Series(LSF.LSF(dataset))

#%%
label = []
for i in range(score.shape[0]):
    if(score.iloc[i]>=1):
        label.append(int(1))
    else:
        label.append(int(0))
        
#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

accuracy = accuracy_score(dataset.label, label)*100
precision = precision_score(dataset.label, label)*100
recall = recall_score(dataset.label, label)*100
confusion_matrix = confusion_matrix(dataset.label, label)