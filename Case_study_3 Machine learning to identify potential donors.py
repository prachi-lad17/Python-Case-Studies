#!/usr/bin/env python
# coding: utf-8

# # Who Do We Target

# * Here we have a dataset of people who we approached for donation for our Election Campaign.
# * In this dataset, we have their information like name, education, income, job, ethnicity
# * Logically, high income people would be best to reach first for donation.

# ### We will built classifier that predicts income level based on their attributes.
# And those will be the persons we will approach first for political donation

# # Step1: Loading data and checking summary statistics.

# In[1]:


## Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


## Loading dataset

file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/adult.data"
adult_data = pd.read_csv(file_name)


# In[3]:


## Checking if data is loaded successfully

adult_data.head()


# In[4]:


## Checking last records

adult_data.tail()


# In[5]:


## statistics

adult_data.describe(include='all')


# In[6]:


## data types of variables

adult_data.dtypes


# In[7]:


## checking column names

adult_data.columns


# In[9]:


## Let's change the column names as given

columns_names = ['age', 'workclass', 'fnlwgt','education','education-num','marital-status','occupation',
                'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country', 'Income']

adult_data = pd.DataFrame(adult_data.values,columns = columns_names)

## Checking the changes

adult_data.head()


# In[10]:


## checking for the missing values.

adult_data.isnull().sum()


# In[11]:


## Checking data types

adult_data.dtypes


# In[12]:


## Every variable should not be an object. 
## We need to fix this problem.
## There is .infer_object() function in pandas which fixes this problem.

adult_data = adult_data.infer_objects()

## checking the changes

adult_data.dtypes


# # Step2: Exploratory Data Analysis (Pre-processing 1)

# In[14]:


adult_data.Income


# In[16]:


## Let's remove space before and after our data.
## Strip() function will help us to remove this space.

adult_data['Income'] = adult_data['Income'].str.strip()

## Checking for the changes

adult_data.head()


# In[27]:


## While checking data I came to know that there is no missing value but "?" is present

adult_data[adult_data['workclass']==" ?"]


# In[26]:


adult_data[adult_data['occupation']==" ?"]


# In[29]:


adult_data[adult_data['native-country']==" ?"]


# In[28]:


## Removing spcl character
## adadult_data = aduadult_data['variable'] ! = (is not equal) " ?"

adult_data = adult_data[adult_data['workclass']!=" ?"]
adult_data = adult_data[adult_data['native-country']!=" ?"]
adult_data = adult_data[adult_data['occupation']!=" ?"]


# In[31]:


adult_data.shape


# In[40]:


## We will drop fnlwgt as it is not significant

adult_data.drop(['fnlwgt'],axis=1,inplace=True)

adult_data.head()


# # Step3: Visualization

# In[41]:


sns.set(style="whitegrid", color_codes=True)
sns.factorplot("sex", col='education', data=adult_data, hue='Income', kind="count", col_wrap=4)


# In[35]:


#Show the relationship between capital loss versus capital gain
sns.relplot(x='education-num',y='Income',data=adult_data)
plt.show()


# In[37]:


adult_data['Income'].value_counts()


# In[39]:


sns.relplot(x='age',y='education',data=adult_data,hue='Income',legend='brief')


# # Step4: Preparing our data for modelling (PreProcessing 2)

# In[42]:


## MinMaxScalar scales the data between 0 and 1, each data point will lie bet 0 and 1

from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Make a copy of the our original df
adultdata_minmax_transform = pd.DataFrame(data = adult_data)

# Scale our numerica data
adultdata_minmax_transform[numerical] = scaler.fit_transform(adultdata_minmax_transform[numerical])

adultdata_minmax_transform.head()


# In[43]:


## Get raw income numbers and drop it from our census_minmax_transform dataframe

income_raw = adultdata_minmax_transform['Income']
adultdata_minmax_transform = adultdata_minmax_transform.drop('Income', axis = 1)


# In[44]:


## One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()

features_final = pd.get_dummies(adultdata_minmax_transform)

# Encode the 'income_raw' data to numerical values
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
income = income_raw.apply(lambda x: 0 if x == "<=50K" else 1)
income = pd.Series(encoder.fit_transform(income_raw))

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

print(encoded)


# In[47]:


adultdata_minmax_transform.nunique()


# # Step5: Splitting data into training and testing

# In[48]:


from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, income, test_size = 0.2, random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# In[49]:


## Total number of records
Total_records =  adult_data.shape[0]

## Number of records where individual's income is more than $50,000
Income_greater_than_50k = adult_data[adult_data['Income']== '>50K'].shape[0]

## Number of records where individual's income is at most $50,000
Income_at_most_50k = adult_data[adult_data['Income'] == '<=50K'].shape[0]

## Percentage of individuals whose income is more than $50,000
prcnt_Income_great50k = Income_greater_than_50k / Total_records *100

## Printing the result
print("Total number of records: {}".format(Total_records))
print("Individuals making more than $50,000: {}".format(Income_greater_than_50k))
print("Individuals making at most $50,000: {}".format(Income_at_most_50k ))
print("Percentage of individuals making more than $50,000: {:.2f}%".format(prcnt_Income_great50k))


# In[50]:


# Calculate accuracy
accuracy = Income_greater_than_50k / Total_records

# Calculating precision
precision = Income_greater_than_50k / (Income_greater_than_50k + Income_at_most_50k)

#Calculating recall
recall = Income_greater_than_50k / (Income_greater_than_50k + 0)

# Calculate F-score using the formula above for beta = 0.5
fscore =  (1  + (0.5*0.5)) * ( precision * recall / (( 0.5*0.5 * (precision))+ recall))

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# In[51]:


## We are going to compare 3 modular codes or classifier so that we don't need to copy and paste all the time.
## That's why we are defining this function.

from sklearn.metrics import fbeta_score, accuracy_score
from time import time


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end - start
        
    #  Get the predictions on the test set,
    #  then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300],predictions_train)
        
    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    
    # Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300],predictions_train,0.5)
        
    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test,predictions_test,0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


# # Model building and comparing classifiers

# In[52]:


# Import the three supervised learning models from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

# Initialize the three models, the random states are set to 101 so we know how to reproduce the model later
clf_A = DecisionTreeClassifier(random_state=101)
clf_B = SVC(random_state = 101)
clf_C = AdaBoostClassifier(random_state = 101)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = int(round(len(X_train) / 100))
samples_10 = int(round(len(X_train) / 10))
samples_100 = len(X_train)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)


# In[53]:


#Printing out the values
for i in results.items():
    print(i[0])
    display(pd.DataFrame(i[1]).rename(columns={0:'1%', 1:'10%', 2:'100%'}))


# In[54]:


from sklearn.metrics import confusion_matrix

plt.figure(figsize=(30,12))

for i,model in enumerate([clf_A,clf_B,clf_C]):
    cm = confusion_matrix(y_test, model.predict(X_test))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize the data

    # view with a heatmap
    plt.figure(i)
    sns.heatmap(cm, annot=True, annot_kws={"size":10}, 
            cmap='Blues', square=True, fmt='.3f')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix for:\n{}'.format(model.__class__.__name__));


# # Results Analysis
# - AdaBoost is the most appropriate for our task.
# 
# - It performs the best on the testing data, in terms of both the accuracy and f-score. 
# - It also takes resonably low time to train on the full dataset, which is just a fraction of the 60 seconds taken by SVM, the next best classifier to train on the full training set. So it should scale well even if we have more data.
# 
# - By default, Adaboost uses a decision stump i.e. a decision tree of depth 1 as its base classifier, which can handle categorical and numerical data. Weak learners are relatively faster to train, so the dataset size is not a problem for the algorithm.

# In[55]:


# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Initialize the classifier
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

# Create the parameters list you wish to tune
parameters = {'n_estimators':[50, 120], 
              'learning_rate':[0.1, 0.5, 1.],
              'base_estimator__min_samples_split' : np.arange(2, 8, 2),
              'base_estimator__max_depth' : np.arange(1, 4, 1)
             }

# Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score,beta=0.5)

# Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters,scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train,y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print(best_clf)


# In[56]:


# Train the supervised model on the training set 
model = AdaBoostClassifier().fit(X_train,y_train)

# Extract the feature importances
importances = model.feature_importances_
importances


# In[57]:


predictions


# In[ ]:




