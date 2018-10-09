# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:47:48 2018

@author: yipin
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
import re
import string
from nltk.stem import WordNetLemmatizer
import os
from os import listdir
from collections import Counter
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import time
from time import time
from sklearn.metrics import classification_report



plt.style.use('ggplot')
os.chdir("C:/Intern/Project")
os.getcwd()



def clean_doc(doc, remove_stopwords= True, output_format="string"):
    wordnet = WordNetLemmatizer()
    doc = re.sub('_', ' ', doc)
    # Remove email
    if doc[0:2] == 'T1' or  doc[0:2] == 'T2':
        flag = True
        pre = doc[0:2]+" "
    else:
        flag = False
    doc = re.sub(r'[\w\.-]+@[\w\.-]+', '', doc)
    ## remove IP address
    doc = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", " IP ",  doc)
    ##--remove URL
    doc = re.sub(r"(hxxp?|https|http)\S+", "URL",  doc)
    ##--remove iadb.org
    doc = re.sub(r"\S+.iadb.org' \S:", " ", doc)
    doc = re.sub(r"\.idb.iadb.org", " ", doc)
    doc = re.sub(r"\.iadb.org", " ", doc)
    ##--remove punctuation
    doc = re.sub(r"[\s+\.\!\/_,|%^*#(+\"\')?<>:-]", " ", doc)
    ##--remove the number
    doc = re.sub(r"[0-9]+", ' ', doc)
    if doc == ' ':
        doc =''
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    #re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    #tokens = [re_punc.sub(' ', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens]
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    if remove_stopwords:
        # Use set as it has O(1) lookup time
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [word for word in tokens if len(word) > 1]
    tokens = [wordnet.lemmatize(word) for word in tokens]
    # Return a cleaned string or list
    if output_format == "string":
        if flag:     
            return (pre + " ".join(tokens))
        else:
            return" ".join(tokens)
    elif output_format == "list":
        if flag:
            return (tokens.insert(0,pre))
        else:
            return(tokens)

def vectorization(X_train, X_test, mode):
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(X_train)
    # encode training data set
    X_train_vec = tokenizer.texts_to_matrix(X_train, mode=mode)
    # encode training data set
    X_test_vec = tokenizer.texts_to_matrix(X_test, mode=mode)
    
    return X_train_vec, X_test_vec, tokenizer



def main():

    df = pd.read_csv('./Data/merge_all_cleaned.csv',encoding = 'cp1252')

    ##-- pick up the correct tickets
    correct = df[df['Q7'] == 1]

    ##--clean the short description
    correct['short_desc_cleaned'] = correct['short_description'].apply(lambda doc: clean_doc(doc)) 
   
    correct['category_id'] = correct['category'].factorize()[0]
    ##--set the prediction variable and the target variable
    X = correct['short_desc_cleaned']
    y = correct['category_id']
    
    
    #--build the dictionary to map category id to category
    category_id_df = correct[['category', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'category']].values)
    
    correct.to_csv('./Data/correct_cleane_all.csv',index = False, encoding = 'cp1252')
    
    mode = 'tfidf'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_vec, X_test_vec,tokenizer = vectorization(X_train, X_test, mode = mode)
    
    ##--Multinomial Naive Bayes Model
    clf = MultinomialNB().fit(X_train_vec,  y_train)
    
    sample = ['Disk Free Space low Server','SAP Issues Traceability Report']
    sample_tfidf = tokenizer.texts_to_matrix(sample, mode=mode)
    predicted = clf.predict(sample_tfidf)
    for i in range(len(sample)):
        print(sample[i])
        print(id_to_category[predicted[i]])
    
    predicted = clf.predict(X_test_vec)
    np.mean(predicted == y_test) 
    
    ##--SVM
    clf2 = SGDClassifier(loss='hinge', penalty='l2',
                                          alpha=1e-3, random_state=42,
                                          max_iter=5, tol=None).fit(X_train_vec,  y_train)
    predicted2 = clf2.predict(X_test_vec)
    np.mean(predicted2 == y_test) 
    
    print(metrics.classification_report(y_test, predicted2))
    
    from sklearn.model_selection import GridSearchCV
    param_grid =  {'loss': ['hinge','perceptron', 'squared_loss'], 'penalty': ['l2','l1','elasticnet'],
           'alpha':[0.01,0.001],'max_iter':[5,10]}
    
    clf2 = GridSearchCV(SGDClassifier(), param_grid)
    clf2.fit(X_train_vec, y_train)
    print("Best parameters are: ", clf2.best_params_)
    print("mean_test_score", clf2.cv_results_['mean_test_score'])
    print("std_test_score", clf2.cv_results_['std_test_score'])
    y_true, y_pred = y_test, clf2.predict(X_test_vec)
    print(classification_report(y_true, y_pred))
    np.mean(y_pred == y_test) 
    
