# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:41:32 2018

@author: yipin
"""
import pandas as pd
import numpy as np
import os
##--read data
os.chdir("C:/Intern/Project/Final")
df = pd.read_csv('one_year.csv',)
df = df[df['category']!= 'Finance Supported Applications']
df = df.reset_index(drop = True)

from datetime import datetime
##--filter out the period that is used for training the model
start_time = datetime(2017, 9, 30)
df['sys_created_on'] = df['sys_created_on'].apply(lambda date: datetime.strptime(date, '%m/%d/%Y %H:%M'))
df = df[(df['sys_created_on'] > start_time) ].reset_index(drop = True)

##-- filter out columns we need
df = df[['category', 'subcategory', 'u_component',
       'short_description_cleaned']]
df['subcategory'] = df['category'] + '__' + df['subcategory']


##--clean the data
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string
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

df['short_description_cleaned'] =df['short_description_cleaned'].apply(lambda doc: clean_doc(doc))
df = df[df['short_description_cleaned'] != '']

## Set the parameters
MAX_FEATURES = 5000
#N_TOPICS = 10
NMF_ALPHA = 0.1
NMF_L1_RATIO = 0.25
N_TOPIC_WORDS = 1
CATEGORY = 'subcategory'
NGRAM_RANGE = (3,4)
VECTORIZER = 'tfidf'
TOPIC_MODEL = 'lda'
PARAMETERS = {'n_components': [5], 'learning_decay': [ .7]}


## text vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
def fit_tfidf(documents):
    tfidf = TfidfVectorizer(input = 'content', stop_words = 'english', use_idf = True, ngram_range = NGRAM_RANGE,
                                    lowercase = True, max_features = MAX_FEATURES, min_df = 1 )
    tfidf_matrix = tfidf.fit_transform(documents.values).toarray()
    tfidf_feature_names = np.array(tfidf.get_feature_names())
    tfidf_reverse_lookup = {word: idx for idx, word in enumerate(tfidf_feature_names)}
    return tfidf_matrix, tfidf_reverse_lookup, tfidf_feature_names

from sklearn.feature_extraction.text import CountVectorizer
def fit_bow(documents):
    bow = CountVectorizer(input='content', ngram_range= NGRAM_RANGE, stop_words='english', min_df=2)
    bow_matrix = bow.fit_transform(documents.values).toarray()
    bow_feature_names = np.array(bow.get_feature_names())
    bow_reverse_lookup  =  {word: idx for idx, word in enumerate(bow_feature_names)}
    return bow_matrix, bow_reverse_lookup, bow_feature_names


def vectorization(documments):
    if VECTORIZER == 'tfidf':
        vec_matrix, vec_reverse_lookup, vec_feature_names = fit_tfidf(documents) 
    if VECTORIZER == 'bow':
        vec_matrix, vec_reverse_lookup, vec_feature_names = fit_bow(documents)
    return vec_matrix, vec_reverse_lookup, vec_feature_names

## explore categories
category_count = pd.DataFrame(df[CATEGORY].value_counts())
category_count = category_count.rename(columns = {CATEGORY:"counts"})
#print the top 10 subcategories
category_count[0:10]

category_list = category_count[(category_count['counts'] >= 30)].index.tolist()
print(category_list)
print(len(category_list))


## Topic Modeling
from sklearn.decomposition import NMF
def nmf_model(vec_matrix, vec_reverse_lookup, vec_feature_names, NUM_TOPICS):
    topic_words = []
    nmf = NMF(n_components = NUM_TOPICS, random_state=3).fit(vec_matrix)
    for topic in nmf.components_:
        word_idx = np.argsort(topic)[::-1][0:N_TOPIC_WORDS]
        topic_words.append([vec_feature_names[i] for i in word_idx])
    return topic_words

documents = df[df[CATEGORY] == "User Non-Standard Software__Skype"]['short_description_cleaned']
vec_matrix, vec_reverse_lookup, vec_feature_names = vectorization(documents)
topic_words = []
nmf = NMF(n_components = 5, random_state=3).fit(vec_matrix)
idx_index = []
for topic in nmf.components_:
    word_idx = np.argsort(topic)[::-1][0:N_TOPIC_WORDS]
    topic_words.append([vec_feature_names[i] for i in word_idx])
    idx_index.append(word_idx)


## Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
def lda_model(vec_matrix, vec_reverse_lookup, vec_feature_names, PARAMETERS):
    topic_words = []
    lda = LatentDirichletAllocation()
    model = GridSearchCV(cv = 3, estimator = lda, param_grid = PARAMETERS)
    model.fit(vec_matrix)
    # Best Model
    best_lda_model = model.best_estimator_
    # Model Parameters
    print("Best Model's Params: ", model.best_params_)
    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)
    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(vec_matrix))

    for topic in best_lda_model.components_:
        word_idx = np.argsort(topic)[::-1][0:N_TOPIC_WORDS]
        topic_words.append([vec_feature_names[i] for i in word_idx])
    return topic_words

table = pd.DataFrame(columns = ['subcategory', 
                                'topic 1','topic 1 weight', 
                                'topic 2','topic 2 weight',
                               'topic 3','topic 3 weight',
                               'topic 4', 'topic 4 weight',
                               'topic 5', 'topic 5 weight'])

j = 0
TOPIC_MODEL = 'nmf'
for category in category_list:
    print("\n The subcategory is %s: \n" % category)
    documents = df[df[CATEGORY] == category]['short_description_cleaned']
    
    #topic modeling
    vec_matrix, vec_reverse_lookup, vec_feature_names = vectorization(documents)
    if TOPIC_MODEL == 'lda':
        topic_words = lda_model(vec_matrix, vec_reverse_lookup, vec_feature_names, PARAMETERS)
    if TOPIC_MODEL == 'nmf':
        topic_words = nmf_model(vec_matrix, vec_reverse_lookup, vec_feature_names, 5)
    for t in range(len(topic_words)):
        print("Topic {}: {}".format(t+1, ', '.join(topic_words[t][:])))
    
    table.loc[j,'subcategory'] = category
    counter = [0,0,0,0,0]
    documents = documents.values
    total_number = len(documents)
    for i in range(len(documents)):
        if topic_words[0][0] in documents[i]:
            counter[0] += 1
        if topic_words[1][0] in documents[i]:
            counter[1] += 1
        if topic_words[2][0] in documents[i]:
            counter[2] += 1
        if topic_words[3][0] in documents[i]:
            counter[3] += 1
        if topic_words[4][0] in documents[i]:
            counter[4] += 1
    for i in range(len(topic_words)):
        topic_name = 'topic' + ' ' + str(i+1)
        weight_name = topic_name + ' ' + 'weight'
        table.loc[j,topic_name] = topic_words[i][0]
        table.loc[j,weight_name] = counter[i]/total_number
    j = j + 1


documents = df[df[CATEGORY] == "Network Devices__Appliance"]['short_description_cleaned']

#topic modeling
vec_matrix, vec_reverse_lookup, vec_feature_names = vectorization(documents)
if TOPIC_MODEL == 'lda':
    topic_words = lda_model(vec_matrix, vec_reverse_lookup, vec_feature_names, PARAMETERS)
if TOPIC_MODEL == 'nmf':
    topic_words = nmf_model(vec_matrix, vec_reverse_lookup, vec_feature_names, 5)
for t in range(len(topic_words)):
    print("Topic {}: {}".format(t+1, ', '.join(topic_words[t][:])))

table.loc[j,'subcategory'] = category
counter = [0,0,0,0,0]
documents = documents.values
total_number = len(documents)
for i in range(len(documents)):
    if topic_words[0][0] in documents[i]:
        counter[0] += 1
    if topic_words[1][0] in documents[i]:
        counter[1] += 1
    if topic_words[2][0] in documents[i]:
        counter[2] += 1
    if topic_words[3][0] in documents[i]:
        counter[3] += 1
    if topic_words[4][0] in documents[i]:
        counter[4] += 1
for i in range(len(topic_words)):
    topic_name = 'topic' + ' ' + str(i+1)
    weight_name = topic_name + ' ' + 'weight'
    table.loc[j,topic_name] = topic_words[i][0]
    table.loc[j,weight_name] = counter[i]/total_number
	
	
table.to_csv("Topic.csv", index = False)





