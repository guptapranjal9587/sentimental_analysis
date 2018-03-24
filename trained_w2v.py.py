
# coding: utf-8

# In[5]:


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
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

pos_review = sorted(glob.glob("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/train/pos/*.txt"))
neg_review = sorted(glob.glob("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/train/neg/*.txt"))
print("loading files....")
pos_raw = u""
for book_filename in pos_review:
    #print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename,"r", "utf-8") as book_file:
        pos_raw+= book_file.read()
pos_review = sorted(glob.glob("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/test/pos/*.txt"))
for book_filename in pos_review:
    #print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename,"r", "utf-8") as book_file:
        pos_raw+= book_file.read()
print("loading negative files..")
neg_raw = u""
for book_filename in neg_review:
    #print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename,"r", "utf-8") as book_file:
        neg_raw+= book_file.read()
neg_review = sorted(glob.glob("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/test/neg/*.txt"))
for book_filename in neg_review:
    #print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename,"r", "utf-8") as book_file:
        neg_raw+= book_file.read()

#print(len(neg_raw))
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
pos_sentences = tokenizer.tokenize(pos_raw)
neg_sentences = tokenizer.tokenize(neg_raw)



def sentence_to_wordlist(raw):
    #clean = re.sub("[^a-zA-Z]"," ", raw)
    clean=raw
    words = word_tokenize(clean.lower())
    words = [ps.stem(w) for w in words]
    return words
sentences = []
for _sentence in pos_sentences:
    if len(_sentence) > 0:
        sentences.append(sentence_to_wordlist(_sentence))
for _sentence in neg_sentences:
    if len(_sentence) > 0:
        sentences.append(sentence_to_wordlist(_sentence))

token_count = np.sum([len(sentence) for sentence in sentences])
#  HYPERPARAMETERS
num_features = 300
min_word_count = 3
# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()
#print(num_workers)
# Context window length.
context_size = 5
# Downsample setting for frequent words.
#rate 0 and 1e-5 
#how often to use
downsampling = 1e-3
# Seed for the RNG, to make the results reproducible.
seed = 1
print("files are loaded..")
review_word2vec = w2v.Word2Vec(sg=1,seed=1,workers=num_workers,size=num_features,min_count=min_word_count,window=context_size,sample=downsampling)
review_word2vec.build_vocab(sentences)
words = list(review_word2vec.wv.vocab)
print("training starts.....")
review_word2vec.train(sentences , total_words=token_count, epochs=3)
review_word2vec.save("trained_w2v")
model = w2v.Word2Vec.load("trained_w2v")


# In[10]:


from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
len(stops)

