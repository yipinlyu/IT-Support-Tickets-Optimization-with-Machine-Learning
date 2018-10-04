# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:19:24 2018

@author: YIPINL
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
os.chdir("C:/Intern/Project/Data")

#--read dataset: df1 contains all the tickets from January 2018 to June 2018
df1 = pd.read_csv('./incident_half_year_2018.csv', encoding = 'cp1252')
##-- pickup columns we what to train the model
df1 = df1[['category', 'subcategory', 'u_component','number',
       'short_description']]
##--remove the Finance Supported Applications
df1 = df1[df1['category']!= 'Finance Supported Applications']
df1.shape

## natural language translator
import pandas as pd
import numpy as np
import json
from watson_developer_cloud import LanguageTranslatorV3
import os
os.chdir("C:/Intern/Project/Data")
language_translator = LanguageTranslatorV3(
    version='2018-05-01',
    iam_api_key='')

##--create a new column to record the translation
df1['translation'] = ''

##--use a for loop to record the translations of corresponding short descriptions
for i in range(0,len(df1)):
    translation = language_translator.translate(
    text= df1.loc[i,'short_description'],
    model_id='es-en')
    df1.loc[i,'translation'] = translation['translations'][0]['translation']
    if (i % 100 == 0):
        print(i)
        
df1.to_csv('translated_2018.csv', index = False, encoding = 'utf-8')