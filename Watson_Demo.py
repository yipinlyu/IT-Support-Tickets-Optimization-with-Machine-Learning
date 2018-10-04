# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:32:50 2018

@author: yipin
"""

import json
import jsonpath 
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
import Features, KeywordsOptions, CategoriesOptions, EntitiesOptions, \
SemanticRolesOptions,  ConceptsOptions, MetadataOptions


natural_language_understanding = NaturalLanguageUnderstandingV1(
  username='',
  password='',
  version='')

import os
os.chdir("D:/Intern/Project/")
os.getcwd()

import pandas as pd
df = pd.read_csv('./Data/merge_all_cleaned.csv',encoding = 'cp1252')


import HTMLParser
html_parser = HTMLParser.HTMLParser()



from html.parser import HTMLParser


print(text3)

HTMLParser.feed(text3)

HTMLParser.handle_data(data)

print(html_parser.unescape(text3))

from bs4 import BeautifulSoup
soup = BeautifulSoup(text3)
soup.body.find('div', attrs={'class' : 'container'}).text


len(pd.unique(df['u_component']))

sample = df.loc[0:10,['short_description','number','close_notes','description.1']]


sample['response1'] = ''
sample['response2'] = ''
sample['response3'] = ''
i = 2
for i in range(len(sample)):
    text1 = sample.loc[i, 'short_description']
    text2 = sample.loc[i,'close_notes']
    text3 = sample.loc[i,'description.1']
    response1 = natural_language_understanding.analyze(
       text=text1,
       features=Features(
       keywords=KeywordsOptions(
      sentiment=False,
      emotion=False)))
    response2 = natural_language_understanding.analyze(
       text=text2,
       features=Features(
       keywords=KeywordsOptions(
      sentiment=False,
      emotion=False)))
    response3 = natural_language_understanding.analyze(
       text=text3,
       features=Features(
       keywords=KeywordsOptions(
      sentiment=False,
      emotion=False)))
    keywords1=jsonpath.jsonpath(response1,"$..text")  
    keywords2=jsonpath.jsonpath(response2,"$..text") 
    keywords3=jsonpath.jsonpath(response3,"$..text") 
    
    
    category1 = natural_language_understanding.analyze(
      text = text1,
      features=Features(
     categories=CategoriesOptions()))

    sample.loc[i,'response1'] = response1
    sample.loc[i,'response2'] = response2
    sample.loc[i,'response3'] = response3
    
    
    response = natural_language_understanding.analyze(
      text = text1,
      features=Features(
      concepts=ConceptsOptions(
      limit=3)))
   response = natural_language_understanding.analyze(
  text = text2,
  features=Features(
    entities=EntitiesOptions(
      sentiment=True,
      limit=1)))
   
    response = natural_language_understanding.analyze(
  text=text3,
  features=Features(
    semantic_roles=SemanticRolesOptions()))

import sys
print(sys.modules)


from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
['this', 'is', 'the', 'second', 'sentence'],
['yet', 'another', 'sentence'],
['one', 'more', 'sentence'],
['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
# fit a 2D PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
