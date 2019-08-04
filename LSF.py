# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:17:18 2018

@author: Latifah
"""
import pandas as pd

data = pd.read_csv("prepros(01).csv", encoding='utf-8')
#%%
def LSF(data):
    taglist = pd.read_csv ("POSTag.csv")
    taglist = pd.Series(taglist.tag.values,index=taglist.word.values).to_dict()
            
    offensive_word = open("offensive_word_add.txt").readlines()
    offensive_word = [s.replace ('\n', '') for s in offensive_word]
            
    pattern_tag = open("pattern_tag.txt").readlines()    
    pattern_tag = [s.replace ('\n', '') for s in pattern_tag]
                
    Os = 0
    Ow = 0.5
    Iw = 2
            
    postag = []
    tag = []
               
    for tweet in data.tweet:  
        for word in tweet.split():        
            if word in taglist:
                if word in offensive_word:
                    tag.append("offensive")
                else:
                    tag.append(taglist[word])                    
            else:
                tag.append('--')
        postag.append(tag)
        tag = [] 
            
            
    sentences = []
            
    for i in range(len(postag)):
        sentences.append(' '.join(postag[i]))
            
    
    #labels
    Os_total = []
        
    for sentence in sentences:
        Ow_count = sentence.count("offensive")
        Iw_count = 0
        #if "offensive" in sentence:    
        for pattern in pattern_tag:            
            if pattern in sentence:                
                Iw_count += 1
                                
        Os = (Ow_count - Iw_count) * Ow
        Os =  Os + Iw_count*(Ow*Iw)
        #if (total_Os >= 1):
            #   kategori = 'B'        
            #  labels.append(kategori)        
        #else:
        #   kategori = 'N'
        #  labels.append(kategori)
        Os_total.append(float(Os))    
        
        Os = 0
        Ow = 0.5
        Iw = 2
    
    return (Os_total)

