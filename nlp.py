# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:47:48 2018

@author: yipin
"""

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
import re
import string
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
os.chdir("D:/Intern/Project/")
os.getcwd()



def clean_doc(doc, remove_stopwords= True, output_format="string"):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    if remove_stopwords:
        # Use set as it has O(1) lookup time
        stops = set(stopwords.words("english"))
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [word for word in tokens if len(word) > 1]

    # Return a cleaned string or list
    if output_format == "string":
        return " ".join(tokens)
    elif output_format == "list":
        return tokens


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
    
    ##K-NN



def vectorization(X_train, X_test, mode):
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(X_train)
    # encode training data set
    X_train_vec = tokenizer.texts_to_matrix(X_train, mode=mode)
    # encode training data set
    X_test_vec = tokenizer.texts_to_matrix(X_test, mode=mode)
    
    return X_train_vec, X_test_vec

    a = tokenizer.texts_to_matrix(correct.loc[1:2,'short_desc_cleaned'].values.tolist(), mode=mode)

    ##plot the first level category
    correct['category_id'] = correct['category'].factorize()[0]
    fig = plt.figure(figsize=(8,6))
    correct.groupby('category').short_desc_cleaned.count().plot.bar(ylim=0)
    plt.show()
    
    ##tf-idf
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(2, 3), stop_words='english')
    features = tfidf.fit_transform(correct.short_desc_cleaned).toarray()
    labels = correct.category_id
    features.shape
    
   
    category_id_df = correct[['category', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'category']].values)
    
    N = 2
    for Product, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 3]
        print("# '{}':".format(Product))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        print("  . Most correlated trigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
    
    X_train, X_test, y_train, y_test = train_test_split(correct['short_desc_cleaned'], correct['category_id'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    modelNB = MultinomialNB()
    clf = modelNB.fit(X_train_tfidf, y_train)
    modelNB.score(X_test, y_test)
    print(clf.predict(count_vect.transform(["RFC Connection Issue"])))
    
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
        ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    
    import seaborn as sns
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    
    
    model = LinearSVC()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, correct.index, test_size=0.33, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    
    cv_df.groupby('model_name').accuracy.mean()
    
    from IPython.display import display
    for predicted in category_id_df.category_id:
        for actual in category_id_df.category_id:
            if predicted != actual and conf_mat[actual, predicted] >= 10:
                print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
                display(correct.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['category', 'short_desc_cleaned']])
                print('')
    
    