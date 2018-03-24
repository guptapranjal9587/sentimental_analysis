
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function
import codecs
#finds all pathnames matching a pattern, like regex
import glob
#log events for libraries
import logging
#concurrency
import multiprocessing
#dealing with operating system , like reading file
import os
#pretty print, human readable
import pprint
#regular expressions
import re
#natural language toolkit
import nltk
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
#word 2 vec
import gensim.models.word2vec as w2v
#dimensionality reduction
import sklearn.manifold
#math
import numpy as np
#plotting
import matplotlib.pyplot as plt
#parse dataset
import pandas as pd
#visualization
import seaborn as sns
#print("dsgafgfag")
pos_review = sorted(glob.glob("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/train/pos/*.txt"))
neg_review = sorted(glob.glob("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/train/neg/*.txt"))


# In[ ]:



    


# In[207]:


from os import listdir
from os.path import isfile, join
data=[]
def label_the_docs(fold,sign):
    doclabels = []
    docnames = []
    doclabels =[(fold+f) for f in listdir("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/"+fold+"/"+sign) if f.endswith('.txt')] 
    docnames =[f for f in listdir("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/"+fold+"/"+sign) if f.endswith('.txt')] 
    return docnames,doclabels

doc_names = []
doc_labels = []
data=[]

direct_list = [("train","pos"),("train","neg"),("test","pos"),("test","neg")]

for w in direct_list:
    d_n,d_l = label_the_docs(w[0],w[1])
    doc_names += d_n
    doc_labels += d_l
    print(len(d_n))
    for doc in d_n:
        data.append(codecs.open("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/"+w[0]+"/"+w[1]+"/"+doc,"r","utf-8").read())
len(doc_names)


# In[172]:


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield d2v.TaggedDocument(words=doc.split(),tags=[self.labels_list[idx]])


# In[173]:


from gensim.models import Doc2Vec
import gensim.models.doc2vec as d2v
from gensim.models.deprecated.doc2vec import LabeledSentence
assert gensim.models.doc2vec.FAST_VERSION > -1
LabeledSentece = gensim.models.deprecated.doc2vec.LabeledSentence
from gensim.models.doc2vec import TaggedDocument


# In[181]:


it =LabeledLineSentence(data,doc_labels)
model = gensim.models.Doc2Vec(it,size=300, window=7, min_count=3, workers=4,alpha=0.025, min_alpha=0.025)


# In[183]:


model.save("paragraph_vector")


# In[203]:




