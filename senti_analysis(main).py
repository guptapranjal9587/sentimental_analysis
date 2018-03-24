from evaluation_of_vectors import *

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
#regular expressions
import re
#natural language toolkit
import nltk
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
#word 2 vec
import gensim.models.word2vec as w2v
#math
import numpy as np



#==========================================================================================================================

# loading data
from sklearn.datasets import load_svmlight_file
import numpy as np
from scipy.sparse import csr_matrix
X, y = load_svmlight_file("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/train/labeledBow.feat")
X_test, y_test = load_svmlight_file("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/test/labeledBow.feat")

# y stores label
y=[1]*(12500)
y=y+[0]*(12500)
y_test=[1]*(12500)
y_test=y_test+[0]*(12500)


#=====================================================================================================

# Normalized term frequency vector
length_X = csr_matrix(1/X.sum(axis=1))
tf_train = X.multiply(length_X)
lenth_X = csr_matrix(1/X_test.sum(axis=1))
tf_test = X_test.multiply(lenth_X)
evaluate(tf_train,y,tf_test,y_test)
NaiveBays(bow_train,y,bow_test,y_test)



#===================================================================================================

# Evaluation with BOW vector representation
bow_train = X.sign()
bow_test = X_test.sign()
evaluate(bow_train,y,bow_test,y_test)
NaiveBays(bow_train,y,bow_test,y_test)


#=================================================================================================


# tf-idf (term frequency inverse document frequency)
def tfidf_vectorizer(X):
    S=[]
    for i in range(89527):
        if(i%1000==0):
            print(i,"th iteration")
        S.append(len(X.getcol(i).nonzero()[0]))
    word_in_doc = csr_matrix(S)
    inv_word_in_doc = word_in_doc.multiply(1/25000)
    inv_word_in_doc=inv_word_in_doc.log1p()
    tf_idf = X.multiply(inv_word_in_doc)
    return tf_idf
# vectorization    
tfidf_train = tfidf_vectorizer(X)
tfidf_test = tfidf_vectorizer(X_test)

# evaluate tfidf
evaluate(tfidf_train,y,tfidf_test,y_test)






#==================================================================================================================
#creatin glove and word2vec



# Word2Vec from gensim (model is trained on given reviews, code is attached for Word2vec training)
model_w2v = w2v.Word2Vec.load("trained_w2v")
words = list(model_w2v.wv.vocab)
words_set = set(words)


from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/glove.6B/glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'glove.6B.100d.txt.word2vec'
model_glove = KeyedVectors.load_word2vec_format(filename, binary=False)
words_glove = set(model_glove.vocab)




from nltk.corpus import stopwords
from bs4 import BeautifulSoup
def sentence_to_wordlist(raw):
    #clean=raw
    word = raw.lower().split()
    #word = [ps.stem(w) for w in word]
    #clen = BeautifulSoup( raw ,"lxml").get_text()
    #clean = re.sub("[^a-zA-Z]"," ", clen)
    # word = clean.lower().split()
    #stops = set(stopwords.words("english"))
    #word=[w for w in word if not w in stops ]
    return word
def create_vect(review,vector_type):
    tokens = sentence_to_wordlist(review)
    if(vector_type == "word2vec"):
        vect = np.sum([model_w2v.wv.__getitem__(stri) for stri in tokens if stri in words_set], axis=0 )/len(tokens)
    else:
        vect = np.sum([model_glove[stri] for stri in tokens if stri in words_glove], axis=0 )/len(tokens)
    return vect

def vectorize_w2v_glove(fold, vector_type):
    pos_review = sorted(glob.glob("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/"+fold+"/pos/*.txt"))
    neg_review = sorted(glob.glob("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/"+fold+"/neg/*.txt"))
    X_train=[] 
    i=0
    print("vectorizing positive "+fold+" reviews for "+vector_type+".....")
    for book_filename in pos_review:
        #print("Reading '{0}'...".format(book_filename))
        with codecs.open(book_filename,"r", "utf-8") as book_file:
            pos_raw = u""
            pos_raw= book_file.read()
            X_train.append( create_vect(pos_raw,vector_type) )
    
    print("vectorizing negative "+fold+" reviews for "+vector_type+".....")
    for book_filename in neg_review:
        #print("Reading '{0}'...".format(book_filename))
        with codecs.open(book_filename,"r", "utf-8") as book_file:
            neg_raw = u""
            neg_raw= book_file.read()
            X_train.append( create_vect(neg_raw,vector_type) )

    return X_train



#=========================================================================================================


#weighted W2V with tfidf

from scipy.sparse import find
data = codecs.open("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/imdb.vocab","r","utf-8").read()
vocab = data.split()
w2v_tfidf_train=[]
w2v_tfidf_test=[]
for i in range(25000):
    rows, cols, value = find(tfidf_train[i])
    vect = np.sum([model_w2v.wv.__getitem__(vocab[j]) for j in cols if vocab[j] in words_set], axis=0 )/np.sum(X[i])
    w2v_tfidf_train.append(vect)
for i in range(25000):
    rows, cols, value = find(tfidf_test[i])
    vect = np.sum([model_w2v.wv.__getitem__(vocab[j]) for j in cols if vocab[j] in words_set], axis=0 )/np.sum(X[i])
    w2v_tfidf_test.append(vect)




# word2vec test_train vectorization
w2v_train = vectorize_w2v_glove("train","word2vec")
w2v_test = vectorize_w2v_glove("test","word2vec")

# word2vec evaluation
evaluate(w2v_train,y,w2v_test,y_test)
# evaluating weighted tfidf w2v
evaluate(w2v_tfidf_train,y,w2v_tfidf_test,y_test)


# ======================================================================================================



# creating wieghted tfidf global vectors
glove_tfidf_train=[]
glove_tfidf_test=[]
for i in range(25000):
    rows, cols, value = find(tfidf_train[i])
    vect = np.sum([model_glove[vocab[j]] for j in cols if vocab[j] in words_glove], axis=0 )/np.sum(X[i])
    glove_tfidf_train.append(vect)
for i in range(25000):
    rows, cols, value = find(tfidf_test[i])
    vect = np.sum([model_glove[vocab[j]] for j in cols if vocab[j] in words_glove], axis=0 )/np.sum(X[i])
    glove_tfidf_test.append(vect)


# glove test_train vectorization
glove_train = vectorize_w2v_glove("train","glove")
glove_test = vectorize_w2v_glove("test","glove")

evaluate(glove_train,y,glove_test,y_test)

# tfidf wieghted glove vector
evaluate(glove_tfidf_train,y,glove_tfidf_test,y_test)


#=========================================================================================================



# paragraph vector for reviews
# loading trained paragraph vectors(code for training is attached)
import gensim.models
from gensim.models import Doc2Vec
import gensim.models.doc2vec as d2v
from gensim.models.deprecated.doc2vec import LabeledSentence
assert gensim.models.doc2vec.FAST_VERSION > -1
model_pv = d2v.Doc2Vec.load("paragraph_vector")




# para graph vectors
from os import listdir
def label_the_docs(fold,sign):
    doclabels = []
    doclabels =[(fold+f) for f in listdir("C:/Users/Pranjal DADA/Desktop/6th sem/NLP/course/aclImdb/"+fold+"/"+sign) if f.endswith('.txt')] 
    return doclabels

doc_labels = []
direct_list = [("train","pos"),("train","neg"),("test","pos"),("test","neg")]
for w in direct_list:
    d_l = label_the_docs(w[0],w[1])
    doc_labels += d_l
    
pv_train =model_pv[doc_labels[:25000]]
pv_test = model_pv[doc_labels[25000:]]  

evaluate(pv_train,y,pv_test,y_test)

