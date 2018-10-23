*IT-support-tickets-optimization-with-machine-learning*


**Problem and Objectives**

A potential problem in the current tickets system is that ticket descriptions
might be assigned to the incorrect categories. Before I start, I would first
describe the general scenario: a staff or a customer would dial “1111” when
he/she is going to report a certain technical problem concerning IT systems,
services or hardware. After acquiring information from the caller, the analyst
will assign the problem to a relevant category and create a ticket which will
then be sent to a corresponding team. In some cases, a ticket would go to the
wrong team: for example, a ticket which should have been categorized as a
telephone problem is sent to the Windows 10 support team which can do nothing
but sending back the problem. Consequently, it would take some additional days
to solve the problem and waste the human resources. Therefore, we need to find
out some solutions to optimize the present tickets system. Primarily, the
project should mainly cover the following three objectives:

(1) Used NB, SVM and LSTM to classify these tickets to different categories.


(2) Used IBM Watson to to the same work.Examined the output generated from the IBM Watson and analyze the false assignment. Adaptd the existing code and train the machine learning models.


(3) Used the topic modeling to extract topics of each categories to support the training of analysts.

** Data Overview**

The dataset I collected were date from September 2017 to July 2018. I did not
include tickets before September 2017 since the linguistic patterns of these
tickets are strikingly distinct from the selected data. 

**1. Main Framework**

![Test Image 4](https://github.com/yipinlyu/IT-support-tickets-optimization-with-machine-learning/blob/master/framework.png)
**Figure 1. The Process of Classification using the IBM Watson**

The Fig. 1 illustrates the overall framework of Project No.1.

1.  Translation: A small number of tickets were written in Spanish. However, the
    IBM Watson Natural Language Classifier can only process one language at one
    time. Hence, I used the IBM Watson Language to convert Spanish to English to
    generate the input data for the next steps.

2.  Data Processing: The process of data processing consists of two parts. The
    first step is Downsampling. I used Downsampling because the dataset is very
    much imbalanced. In 151 subcategories, 29 subcategories just have one
    tickets. Meanwhile, 35 subcategories have more than 50 tickets. If we
    conducted random sampling towards all the subcategories, then we would face
    a problem that we might miss all the tickets in some categories. Hence, I
    conducted sampling towards subcategories that have more than ten tickets.
    The second part is data cleaning. We need to remove some unusual characters,
    such as “\>\>”, otherwise we will meet the “training failure” problem for
    the IBM Watson.

3.  Training the model: After sending the data to the IBM Watson Bluemix Natural
    Language Classifier API, I got a trained model. Successively, we could use
    this model to make predictions of subcategories based on the short
    descriptions. In comparison, I also used the Scikit-Learn and Keras in
    Python to train some machine learning models to make predictions.[1] [2]

4.  Model Evaluation: I used the tickets that had been reviewed by the quality
    assurance team to test the accuracy of this model. Here I adopted two
    metrics to evaluate the accuracy. The first is the accuracy that the No. 1
    prediction (prediction with the highest probability) is equal to the actual
    subcategory in this ticket. I named this metric as the Top 1 Accuracy. The
    second is the accuracy that the real subcategory is in the three most
    possible predictions. I named this metric as the Top 3 Accuracy. The target
    accuracy for the Top 1 Accuracy is 75%, and 80% for the Top 3 Accuracy.

5.  Make predictions of new tickets and conduct Analysis with the output of the
    model: I used this model to make predictions to the categories and
    subcategories for tickets in a recent month.

**2 Model Evaluation**

**Table 1. The Accuracy of Different Models**

| Model                        | Subcategory                   |                        |       |
|------------------------------|-------------------------------|------------------------|-------|
|                              | BOW                           | TF-IDF                 |       |
| Traditional Machine Learning | Naïve Bayes                   | 0.625                  | 0.675 |
|                              | KNN                           | 0.543                  | 0.517 |
|                              | Linear SVM                    | 0.709                  | 0.659 |
|                              | Random Forest                 | 0.651                  | 0.655 |
| Neural Network               | Long Short-Term Memory (LSTM) | 0.628                  |       |
| IBM Watson                   | Natural Language Classifier   | 0.768 (Top 1 Accuracy) |       |
|                              |                               | 0.867 (Top 3 Accuracy) |       |

As table 1 shows, the Top 1 Accuracy of the model is 76.8%, and the Top 3
Accuracy of the model is 86.7%. This means that the IBM Watson outperformed
other models in this case. Hereby I consider that the Watson Bluemix implements
some complex artificial neural network to train the model, because training a
model cost about half a day. Considering we have 151 subcategories in this
dataset, the Top 3 Accuracy is high enough for us to apply in the business
domain.

As a result, more than 81.4% of them fell in correct categories. This
indicated that we could trust the support teams in general. However, there still
existed two problems: (1) overlapping subcategories and (2) ambiguous short
description. 

**3 Topic Modeling**

The previous sections mainly illustrated supervised learning. In this section, I
will present the process of Topic Modeling which consists of three parts: (1)
deep data cleaning, (2) text vectorization, and (3) running topic modeling
algorithms[4] [5] .

First, I conducted deep data cleaning otherwise there would be
numerous useless words and characters in the output data set. The first step I
adopted was to replace the “IP address” with "ip", and to remove non-linguistic
patterns, e. g. punctuation, numbers, email addresses, URLs, etc. Next, I used
the NLTK package in Python to tokenize the short descriptions and ran the
WordNetLemmatizer function to do lemmatization. Successively, I removed the stop
words and generated the cleaned data set.

Second, I converted these texts to a matrix by using the Bag of Words (BOW) and
the Term Frequency-Inverse Document Frequency (TF-IDF). I adopted the 3-gram and
4-gram because three-word or four-lexical bundles contain much more information
than two-word and one-word lexical bundles.

Finally, I applied the Non-negative Matrix Factorization (NMF) and Latent
Dirichlet allocation (LDA) to generate the top five topics of each subcategory.
This document can assist the Quality Assurance team to identify the pattern of
phrases for each category, and then assist the training of new analysts.

 References:

[1] 	Altintas, M., & Tantug, A. C. (2014). Machine learning based ticket classification in issue tracking systems. In Proceedings of International Conference on Artificial Intelligence and Computer Science (AICS 2014).

[2] 	Brownlee, J. (2018). Long Short-Term Memory Networks With Python. Retrieved from https://machinelearningmastery.com/lstms-with-python/.

[3] 	Brownlee, J. (2018). Deep Learning for Natural Language Processing. Retrieved from https://machinelearningmastery.com/lstms-with-python/.

[4] 	Eckstein, L., Kuehl, N., & Satzger, G. (2016, August). Towards Extracting Customer Needs from Incident Tickets in IT Services. In Business Informatics (CBI), 2016 IEEE 18th Conference on (Vol. 1, pp. 200-207). IEEE. 

[5] 	Beneker, D., & Gips, C. (2017). Using Clustering for Categorization of Support Tickets.

