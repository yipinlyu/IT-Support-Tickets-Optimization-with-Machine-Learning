**Internship Report**

by

Lu, Yipin

M.S. in Analytics, Georgetown University

Institution: Inter-American Development Bank

Supervisor: Vasconcellos Jr., Paulo E-mail: PAULOV\@iadb.org

Date: June 16, 2018 – August 15, 2018

1.  **Introduction**

This summer, I served as an intern for the Information Technology Department
(ITE), the Inter-American Development Bank. During the internship, I fulfilled
two projects which required the expertise of machine learning, Natural Language
Processing (NLP) and data visualization: Project No.1, assisted by IBM Watson
and Skicit-Learn, was to optimize the current IT support tickets system; Project
No.2, by applying Tableau and R, was to visualize the relationship among vendors
and staffs. At the end of the internship, I made a presentation on the projects
I have done to Mr. Leandro Corte, the chief of ITE. Mr. Corte generously
provided me with several suggestions, which I will continue working on in
prospect.

1.  **Project No. 1 Tickets Optimization with IBM Watson and Machine Learning:**

**2.1 Problem and Objectives**

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

1.  Integrate the Machine Learning techniques in the IBM Watson Natural Language
    Classifier to automatically assign IT Support tickets to some categories.

2.  Examine the output generated from the IBM Watson and analyze the false
    assignment. Adapt the existing code and train the machine learning models.

3.  Use the trained models to re-entitle categories to support the training of
    analysts.

**2.2 Data Overview**

The dataset I collected were date from September 2017 to July 2018. I did not
include tickets before September 2017 since the linguistic patterns of these
tickets are strikingly distinct from the selected data. The Example. 1 is a
sample randomly drawn from the dataset:

**Example. 1: Sample ticket**

*Short Description: Outlook \> \> Mailbox \> \> User is not receiving emails in
his/her mailbox*

*Category: User Standard Software*

*Subcategory: MS Outlook*

There are 13 categories and 151 subcategories in the dataset in totality.
According to the scopes they covered, the dataset consists of two parts. The
first part consists of 2,697 tickets that have been evaluated by the Quality
Assurance Team which had been categorized correctly. The second part consists of
21,403 tickets which had not been reviewed by the Quality Assurance (QA) Team.
The QA team cannot analyze all the tickets because of the lack of human
resource. What they do is to do random sampling to select some tickets and
review them monthly. The number of tickets that have been reviewed by them is
small compared to all the tickets. Hence, I used the second part to train the
models and used the first part to test and verify the correctness of the output.

**2.3 Main Framework**

![](media/0b56b9b8af539679ba18b8f214d45419.png)

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

**2.3 Model Evaluation**

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

**2.4 Discovery and Recommendations**

I applied this model on 3,300 tickets in July, and I used the Top 3 Accuracy as
the evaluation metric. I did not choose the Top 1 Accuracy because of the
problem of overlapping categories, and I will discuss this aspect later in this
article.

![](media/37ad29d7131f69710cc6a57dde83b90b.png)

**Figure 2. Ticket Analysis Result**

As Fig. 2 shows, at least 81.4% of them fell in correct categories. This
indicated that we could trust the support teams in general. However, there still
existed two problems: (1) overlapping subcategories and (2) ambiguous short
description. Firstly, two samples of overlapping subcategories were presented in
the following (see Example. 2 and Example. 3):

**Example. 2:**

*Short Description: TL - IP Desk Phone\> not registered*

*Category: Telecom Devices Subcategory: VoIP*

**Example. 3:**

*Short Description: IP Desk Phone \>\> Registering*

*Category: End User Devices Subcategory: Telephones*

The problem of overlapping categories has been illustrated in Example. 2 and
Example. 3. Both two tickets described the problem concerning registration of IP
Desk Phone. However, one was assigned to the VoIP subcategory, and the other was
assigned to the Telephones subcategory. Although both analysts were correct, the
overlaps between these two categories might lead to the wrong classification.
There are a lot of tickets that have this problem. Hence, we need to mitigate
the redundancy, such as removing the Telephones subcategory, or train the
analysts to clarify the differences among different categories.

**Example. 4:**

*Short Description: SW - Software Issue*

**Example. 5:**

*Short Description: SAP - Software Issue (experiencing problems in SAP finance,
systems)*

Secondly, the Example. 4 was ambiguous short descriptions. This suggested that
we may attribute this problem to the competence of the software which fell short
in processing accurate prediction. Hence, it is necessary to train these
analysts to use standard templates, such as the short description in the Fig. 6.

**2.5 Topic Modeling**

The previous sections mainly illustrated supervised learning. In this section, I
will present the process of Topic Modeling which consists of three parts: (1)
deep data cleaning, (2) text vectorization, and (3) running topic modeling
algorithms[4] [5] .

First, the requirement for data cleaning is different from that described in
Section 2.3. We need to conduct deeper data cleaning otherwise there would be
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

**Example. 6 Topic modeling for VoIP subcategory**

*Category: Telecom Devices*

*Subcategory: VoIP*

*Topic 1: cisco jabber opening*

*Topic 2: telecom device voip jabber*

*Topic 3: telecom device ip*

*Topic 4: jabber working properly*

*Topic 5: device voip user*

As the Example. 6 shows, the main topics of the VoIP subcategory focus on the
“cisco jabber,” “telecom device” and “voip”. The topic modeling technique will
be very beneficial to train new analysts on how to write a standard ticket to
describe each category.

1.  **Project 2: Visualization the Vendor Relationships**

**3.1 Current Problem and Business Case**

The current strategy for data management concerning vendor relationships is, by
running SQL query, to collect the corresponding vendor’s name and company, team
leader’s name and supervisor’s name. Then, some statistics, such as how many
vendors under a supervisor would be drawn from the collected data. However, this
kind of solution cannot sketch the panorama of the vendor relationships.
Therefore, I, assisted by Tableau and R, made efforts in visualizing the vendor
relationship.

**3.2 Visualization with Tableau**

The Tableau does not have templates for the organizational chart. Hence, I
conducted some researches and found a solution developed by Josh Weyburne[6] . I
will briefly introduce the method below:

1.  Wrangled the data set to create a two-column data frame that described all
    the relationships among these people and company.

2.  Used the self-join and union operation in Tableau to extract the supervisor
    nodes level by level, determined the Y positions, and marked lines to draw.

3.  Used the first character of each word to create an Alphabetic order for each
    node, and thus to determine the X positions.

4.  Created a combined field for each node and its Y position. And then drew the
    lines to make the organizational chart.

![](media/e0b08a4fe5a6a31fc8cd48b8c39db478.png)

![](media/da573a8a11f1d747e533d698e232b9ac.png)

![](media/d4931c2e8b54dac5c028da65e3d163c7.png)

**Figure 3. Organizational Chart Demo (Tableau)**

Because of the privacy concern, I am not likely to show the detailed figures
that contain the full names of vendors, team leaders and supervisors. Hence, as
Fig. 3 indicates, I created a demo of a five-level organizational chart to
present the result. The left figure shows the overall organizational chart for
both a and b. The middle and the right figures illustrate the filtered
organizational chart for a and b. These figures look good and can be updated in
real-time, because I can connect the Tableau to the IBM DB2 Warehouse database
by using the SQL query.

However, we can find the main weakness of this method in Fig. 3. The node d has
two supervisor nodes, which means that some nodes under d’s management work for
a, and some for b. But the middle and the right graphs present all the nodes
under d’s management. The reason is that when I processed the input dataset, I
just kept the relationships between two neighbor levels, to generate the input
which is required by this method. In the real dataset, one company might be
managed by multiple team leaders. Therefore, we cannot use Tableau to draw an
organizational chart if there exist multiple-to-multiple relationships.  
However, the time spent on Tableau is still worthy. From this process, I got to
understand the advanced techniques in Tableau, such as how to conduct self-join
and union operations, create customized calculated field, and customize X and Y
positions for customized graphs. I also came to be aware of the limitations of
Tableau. The coding functions in Tableau cannot support data wangling
operations, such as the append and the split. Hence, it is not appropriate for
us to conduct complicated data wrangling in Tableau. We need to use SQL, R or
Python to preprocess the dataset in advance. Tableau is very powerful when it
has the template that you want.

**3.3 Visualization with R**

I used the networkD3 package in R to visualize the vendor relationships[7] .
This package provides an interface to D3 in java, and it can support two kinds
of organizational charts: The Hyperbolic Tree Chart and the Diagonal Network
Chart. However, it requires a JSON-like input list that describes the
hierarchical relationships. Hence, I developed a code to convert the input data
frame to a JSON-like list by using the unique, append and strsplit function in
R.

![](media/93bbb5f8c80079ec78549fd1712d6272.png)

![](media/59c41bfc0c048cf04bfb0142f43edbd8.png)

**Figure 4. Hyperbolic Tree Chart (Left) and Diagonal Network Chart (Right)**

As Fig. 4 shows, the Hyperbolic Tree Chart gives us an excellent overall
picture, while the Diagonal Network Chart clearly illustrate the hierarchical
relationships. Notice that in this case, the node A has two supervisors. But we
can still visualize the two relationships independently. Therefore, even if we
have four or five levels, this method can still work efficiently to visualize
the vendor relationships correctly.

1.  **Conclusion**

To conclude, my project provides a feasible solution to predict the categories
automatically based on the short descriptions. And the result shows that the
overall performance of support groups is good. However, we need to remove some
redundant categories and train the analysts to distinguish the differences among
categories. I show the topic modeling technique can assist the training of the
analysts. Moreover, I used Tableau and R studio to create a visualization for
vendor management.

**References:**

1.  Altintas, M., & Tantug, A. C. (2014). Machine learning based ticket
    classification in issue tracking systems. In *Proceedings of International
    Conference on Artificial Intelligence and Computer Science (AICS 2014)*.

2.  Brownlee, J. (2018). *Long Short-Term Memory Networks With Python*.
    Retrieved from https://machinelearningmastery.com/lstms-with-python/.

3.  Brownlee, J. (2018). *Deep Learning for Natural Language Processing*.
    Retrieved from https://machinelearningmastery.com/lstms-with-python/.

4.  Eckstein, L., Kuehl, N., & Satzger, G. (2016, August). Towards Extracting
    Customer Needs from Incident Tickets in IT Services. In *Business
    Informatics (CBI), 2016 IEEE 18th Conference on* (Vol. 1, pp. 200-207).
    IEEE.

5.  Beneker, D., & Gips, C. (2017). Using Clustering for Categorization of
    Support Tickets.

6.  Weyburne, J. (2017, February 27). *Building an Organizational Chart in
    Tableau*. Retrieved from
    https://www.dataplusscience.com/SimpleOrgChart.html.

7.  Gandrud,C. Allaire, JJ. Russell,K. and Yetman, CJ. (2017, March 18).
    *networkD3: D3 JavaScript Network Graphs from R*. Retrived from
    http://christophergandrud.github.io/networkD3/.
