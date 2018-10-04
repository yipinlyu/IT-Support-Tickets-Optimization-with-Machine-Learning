# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:41:32 2018

@author: yipin
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
os.chdir("C:/Intern/Project/Final")

import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
os.chdir("C:/Intern/Project/Final")

##--filter out the period that is used for training the model
start_time = datetime(2017, 8, 31)
end_time = datetime(2018,5,1)
df1 = df[(df['sys_created_on'] > start_time) & (df['sys_created_on'] < end_time)].reset_index(drop = True)

##make the training dataset and the testing dataset
def difference_set(df1, df2):
    tickets_to_drop = df2['number'].tolist()
    flag = df1['number'].isin(tickets_to_drop)
    # make flags for difference set
    diff_flag = [not f for f in flag]
    # diff is the differencial set
    diff = df1[diff_flag]
    diff.index = [i for i in range(len(diff))]
    
    #inter is the intersection
    inter = df1[flag]
    inter.index = [i for i in range(len(inter))]
    return diff, inter


df1, df2 = difference_set(df1, df2)
print(df1.shape)
print(df2.shape)

##--count the number of tickets in each subcategory
CATEGORY = 'subcategory'
category_count = pd.DataFrame(df1[CATEGORY].value_counts())
category_count = category_count.reset_index()
category_count = category_count.rename(columns = {CATEGORY:"counts", 'index':CATEGORY})

category_count[0:5]

##--down sampling: make a subset of categories not be removed
import numpy as np
top_count_threshold = 10
minority = np.array(category_count[CATEGORY][category_count['counts']<top_count_threshold].dropna())
print('The minority subcategories are:', minority)

##--split the dataset to majority categories and monority categories
minority_flag = df1[CATEGORY].isin(minority)
majority_flag = [not f for f in minority_flag]
df_minority = df1[minority_flag]
df_majority = df1[majority_flag]

df_majority_downsampled = df_majority.sample(frac=0.9, replace=False, random_state = 3)
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

##--remove the tabs at the end of texts
import re
df_downsampled['short_description_cleaned'] = df_downsampled['short_description_cleaned'].apply(lambda doc: re.sub("\t+$","",doc))
df_downsampled = df_downsampled.reset_index(drop = True)

df_downsampled[['short_description_cleaned',CATEGORY]].to_csv('train_downsampled(201709-201804).csv', encoding = 'utf-8',
                                                          index = False,header = False)
df_downsampled.head(3)

## Train the model
##  API
from watson_developer_cloud import NaturalLanguageClassifierV1
import json
natural_language_classifier = NaturalLanguageClassifierV1(
    username='',
    password=''
)


##with open('train_downsampled(201709-201804).csv', 'rb') as training_data:
    classifier = natural_language_classifier.create_classifier(
    training_data=training_data,
    # before Python SDK v1.0.0, name and language were top-level parameters
    metadata= '{"name": "Classifier_tickets_201709-201804", "language": "en"}'
  )
	
classifiers = natural_language_classifier.list_classifiers()
print(json.dumps(classifiers, indent=2))
status = natural_language_classifier.get_classifier('f8936cx534-nlc-129')
print (json.dumps(status, indent=2))


### Test the model with the testing dataset
df_test = df2
import re
df_test['short_description_cleaned'] = df_test['short_description_cleaned'].apply(lambda doc: re.sub("\t+$","",doc))
df_test.to_csv('test(201709-201704).csv', encoding = 'utf-8', index = False)

#import pandas as pd
df_test = pd.read_csv('test(201709-201704).csv')

## This function gives us the result of predictions using the classifer
def make_prediction(test, classifier_name):
    test_result = pd.DataFrame(columns = ['subcategory', 'number', 'short_description_cleaned', 'prediction'])
    for i in range(0, len(test) - 1, 30):
        df_temp = test[i:i + 30][['subcategory','number','short_description_cleaned']]
        texts = []
        df_temp = df_temp.reset_index(drop = True)
        for j in range(len(df_temp)):
            temp = {'text':df_temp.loc[j,'short_description_cleaned']}
            texts.append(temp)
        classes = natural_language_classifier.classify_collection(classifier_name, 
           texts)
        df_temp['prediction'] = ''
        df_temp['top 3 classes'] = ''
        for j in range(len(df_temp)):
            df_temp.loc[j,'prediction']= classes['collection'][j]['top_class']
            candidate_list = []
            for m in range(3):
                candidate_list.append(classes['collection'][j]['classes'][m]['class_name'])
            df_temp.loc[j,'top 3 classes'] = candidate_list
        test_result = test_result.append(df_temp)
    
    test_result = test_result.reset_index(drop = True)
    test_result['In_top_class'] = test_result['prediction'] == test_result['subcategory']
    
    for i in range(len(test_result)):
        if test_result.loc[i,'subcategory'] in test_result.loc[i, 'top 3 classes']:
            test_result.loc[i,'In_top3'] = True
        else:
            test_result.loc[i,'In_top3'] = False
    test_result = pd.merge(test_result,df[['number','assignment_group','sys_created_on']], on = 'number', how = 'left' ).reset_index(drop = True)
    return test_result


test_result = make_prediction(df_test, '')
test_result.to_csv('test_result(09-04).csv', index = False)

test_result = pd.read_csv("test_result(09-04).csv")

test_result = test_result.rename(columns = {
    'number': 'Number',
    'prediction':'Top 1 Prediction',
    'short_description_cleaned':"Short Description",
    'subcategory': 'Subcategory',
    'top 3 classes': "Top 3 predictions",
    'In_top_class': "In top 1",
    'In_top3': 'In top 3',
    'assignment_group':"Assignment group"    
})
test_result.columns

accuracy1 = test_result['In top 1'].value_counts()[1]/sum(test_result['In top 1'].value_counts())
print("The accuracy for the top one prediction is: %.3f" % accuracy1)

accuracy2 = test_result['In top 3'].value_counts()[1]/sum(test_result['In top 3'].value_counts())
print("The accuracy for the top three predictions is: %.3f" % accuracy2)

CATEGORY = 'Subcategory'
## Conduct analysis towards 
def category_analysis(test_result):
    category_list = list(set(test_result[CATEGORY]))
    analysis = pd.DataFrame(columns = [CATEGORY, 'Number of tickets', 'Accuracy_Top 1','Accuracy_Top 3',
                                       'Number of wrong predictions_Top 1', 'Number of wrong predictions_Top 3'])
    i = 0
    for category in category_list:
        temp = test_result[test_result[CATEGORY] == category][[CATEGORY, 'In top 1','In top 3']]
        n_tickets = len(temp)
        pred_acc1 = len(temp[temp['In top 1']==True])/(len(temp))
        n_fail1 = len(temp[temp['In top 1']==False])
        
        pred_acc2 = len(temp[temp['In top 3']==True])/(len(temp))
        n_fail2 = len(temp[temp['In top 3']==False])
        
        analysis.loc[i, CATEGORY] = category
        analysis.loc[i, 'Number of tickets'] = n_tickets
        analysis.loc[i, 'Accuracy_Top 1'] = pred_acc1
        analysis.loc[i, 'Number of wrong predictions_Top 1'] = n_fail1

        analysis.loc[i, 'Accuracy_Top 3'] = pred_acc2
        analysis.loc[i, 'Number of wrong predictions_Top 3'] = n_fail2
        i = i+1
        #print("Subcategory: %s, Number of tickets: %d, Accuracy: %f" % (category,n_tickets, pred_acc))
    return analysis
#test_category_analysis = category_analysis(test_result)
#test_category_analysis.to_csv('test_analysis_category.csv', index = False)


#failure analysis for each assignment groups
def group_analysis(test_result):
    group_list = list(set(test_result['Assignment group']))
    analysis = pd.DataFrame(columns = ['Assignment group', 'Number of tickets', 'Accuracy_Top 1','Accuracy_Top 3',
                                       'Number of wrong predictions_Top 1', 'Number of wrong predictions_Top 3'])
    i = 0
    for group in group_list:
        temp = test_result[test_result['Assignment group'] == group][['Assignment group','In top 1','In top 3']]
        n_tickets = len(temp)
        pred_acc1 = len(temp[temp['In top 1']==True])/(len(temp))
        n_fail1 = len(temp[temp['In top 1']==False])
        
        pred_acc2 = len(temp[temp['In top 3']==True])/(len(temp))
        n_fail2 = len(temp[temp['In top 3']==False])
        
        analysis.loc[i,'Assignment group'] = group
        analysis.loc[i, 'Number of tickets'] = n_tickets
        analysis.loc[i, 'Accuracy_Top 1'] = pred_acc1
        analysis.loc[i, 'Number of wrong predictions_Top 1'] = n_fail1

        analysis.loc[i, 'Accuracy_Top 3'] = pred_acc2
        analysis.loc[i, 'Number of wrong predictions_Top 3'] = n_fail2
        i = i+1
        
        #print("Subcategory: %s, Number of tickets: %d, Accuracy: %f" % (category,n_tickets, pred_acc))
    return analysis
#test_group_analysis =  group_analysis(test_result)
#test_group_analysis.to_csv('test_analysis_group.csv', index = False)

test_category_analysis = category_analysis(test_result)
test_category_analysis.to_csv('test_analysis_category.csv', index = False)
test_group_analysis =  group_analysis(test_result)
test_group_analysis.to_csv('test_analysis_group.csv', index = False)

## Evaluate the model with new dataset
import pandas as pd
import numpy as np
import json
from watson_developer_cloud import LanguageTranslatorV3
from watson_developer_cloud import NaturalLanguageClassifierV1
import json

language_translator = LanguageTranslatorV3(
    version='',
    iam_api_key='')

natural_language_classifier = NaturalLanguageClassifierV1(
   username='',
    password=''
)

##--change the directory
import os
os.chdir("C:/Intern/Project/Final")

##--read data and remove the Finance Supported Applications
df_eval = pd.read_csv('July.csv', encoding = 'cp1252')
df_eval = df_eval[df_eval['category']!= 'Finance Supported Applications']
df_eval['short_description_cleaned'] = ''
df_eval = df_eval.reset_index(drop = True)

##--translate the short description
for i in range(0,len(df_eval)):
    translation = language_translator.translate(
    text= df_eval.loc[i,'short_description'],
    model_id='es-en')
    df_eval.loc[i,'short_description_cleaned'] = translation['translations'][0]['translation']
    if (i % 100 == 0):
        print(i)
	
import re
df_eval['short_description_cleaned'] = df_eval['short_description_cleaned'].apply(lambda doc: re.sub("\t+$","",doc))
df_eval.to_csv('incidents_July_translated.csv',index = False)

MONTH = 7

df = pd.read_csv("incidents_July_translated.csv")
df['sys_created_on'] = df['sys_created_on'].apply(lambda date: datetime.strptime(date, '%m/%d/%Y %H:%M'))
df['subcategory'] = df['category']+'_'+df['subcategory']
start_time = datetime(2018, MONTH, 1)
end_time = datetime(2018,MONTH+1,1)
df_eval = df[(df['sys_created_on'] > start_time) & (df['sys_created_on'] < end_time)].reset_index(drop = True)

eval_result = make_prediction(df_eval, 'f8936cx534-nlc-129')
eval_result.to_csv('eval_result_month'+str(MONTH)+'.csv', index = False)

eval_result = pd.read_csv('eval_result_month'+str(MONTH)+'.csv')
eval_result = eval_result.rename(columns = {
    'number': 'Number',
    'prediction':'Top 1 Prediction',
    'short_description_cleaned':"Short Description",
    'subcategory': 'Subcategory',
    'top 3 classes': "Top 3 predictions",
    'In_top_class': "In top 1",
    'In_top3': 'In top 3',
    'assignment_group':"Assignment group"    
})

eval_category_analysis = category_analysis(eval_result)
eval_category_analysis.to_csv('eval_analysis_month'+str(MONTH)+'_category.csv', index = False)

print("Number of tickets:", len(eval_result))
accuracy1 = eval_result['In top 1'].value_counts()[1]/sum(eval_result['In top 1'].value_counts())
print("The accuracy for the top one prediction is: %.3f" % accuracy1)

accuracy2 = eval_result['In top 3'].value_counts()[1]/sum(eval_result['In top 3'].value_counts())
print("The accuracy for the top three predictions is: %.3f" % accuracy2)

eval_group_analysis = group_analysis(eval_result)
eval_group_analysis.to_csv('eval_analysis__month'+str(MONTH)+'_group.csv', index = False)
