# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 08:23:05 2018

@author: Latifah
"""

import pandas as pd
import re
import modSpellChecker as sc
from nltk.tokenize import MWETokenizer

data = pd.read_csv("data_testing.csv", encoding='utf-8')
#%%

data['tweet'] = data['tweet'].str.replace('b"','')
data['tweet'] = data['tweet'].str.replace("b'",'')
#%%

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

char = ['.',',',';',':','?','!','(',')','[',']','{','}','<','>','"',
        '/','\'','#','-','@','&','^','~','*']

stop = open('stopword.txt').readlines()

stop = [s.replace ('\n', '') for s in stop]

#%%
slang = pd.read_csv('slangword.csv')

slang = pd.Series(slang.correction.values,index=slang.slang.values).to_dict()

#%%
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

#%%
def token(s):
    return tokens_re.findall(s)
 
def tokenize(s, lowercase=True):
    tokens = token(str(s))
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

#%%
def spellNormalize(s):
    spellCheck = []
    for tokens in tokenize(s):
        if tokens not in char :
            j = sc.correction (tokens)
            spellCheck.append(j)
        else :
            spellCheck.append(tokens)
    return spellCheck

#%%
def Multiword(s):
    mw = [(line.split()[0], line.split()[1]) for line in open('multiword.txt').read().splitlines()]
    tokenizer = MWETokenizer(mw)
    return tokenizer.tokenize(s)

#%%
def slangNormalize(words, slang_dict):    
    for x, y in slang_dict.items() :
       words = words.str.replace ("^"+x+"$", y) 
    return words
   
#%%
        
for i in range(data['tweet'].shape[0]):
   data['tweet'].iloc[i] = [tokens for tokens in spellNormalize(data['tweet'].iloc[i])
                            if tokens not in stop and not tokens.startswith(('http'))]   
   word = data['tweet'].iloc[i]
   multiword = Multiword(word)
   
   multi = pd.Series(multiword)
   slangword = slangNormalize(multi, slang)
   data['tweet'].iloc[i] = " ".join(slangword)
   #data['tweet'].iloc[i] = [tokens for tokens in data['tweet'].iloc[i] 
                           #if tokens not in stop and not tokens.startswith(('#','http'))]
   
   #prepos = data['tweet'].iloc[i]
   #data['tweet'].iloc[i] = [tokens for tokens in tokenize(prepos)
                            #if tokens not in stop and not tokens.startswith(('#','http'))]
                                
#%%
data.to_csv('testing_prepros(01).csv',index=False)

#%%

tes = pd.read_csv('testing_prepros(01).csv')
