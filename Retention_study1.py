
## Step 1 - Loading necessary libraries and datasets.

### Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


### Importing dataset.
hr_data = pd.read_csv(r"F:\Data_science\UDEMY\datasets_datascienceforbusiness\datascienceforbusiness-master\hr_data.csv")

# To view first 5 records we use .head()
hr_data.head()

# To view last 5 records we use .tail()
hr_data.tail()

# To get information about data
hr_data.info()

# To get data types of variables
hr_data.dtypes

#If there are many variables in dataset and it is tidious to find out which variables are categorical by checking it 
# manually then we can find out the categorical variables by below code."""
hr_data.select_dtypes(exclude=['int64','float64']).columns  # """ department and salary are the categorical variables."""

# Display values in categorical columns by .unique()
print(hr_data['department'].unique())
print(hr_data['salary'].unique())

## Size of dataset
hr_data.shape

## ----------------------------------------------------------------------------------------------##

## Step 2 - Loading evaluation and employee satisfaction dataset.

### Importing another file (employee_satisfaction_evaluation)
emp_satisfaction = pd.read_csv(r"F:\Data_science\UDEMY\datasets_datascienceforbusiness\datascienceforbusiness-master\employee_satisfaction_evaluation.csv")

## checking if it is imported 
emp_satisfaction.head()
emp_satisfaction.tail()

### Checking size 
emp_satisfaction.shape  ## There are almost 15k records and 3 columns 

hr_data.shape  ## By checking the shape we know that both dataset have same records hence we can join datasets easily.

## ----------------------------------------------------------------------------------------------##

## Step 3 - Merge and join datasets (table)

emp_satisfaction.head()
hr_data.head()

## Join two data sets by join
main_df = hr_data.set_index('employee_id').join(emp_satisfaction.set_index('EMPLOYEE #'))
main_df = main_df.reset_index()  ### Removing the extra row in the col titles
main_df.head()

## ----------------------------------------------------------------------------------------------##

## Step 4 - EDA on main_df

main_df.tail()

### Checking summary statistics
main_df.describe(include="all")

### checking if there are any missing values or null values
main_df.isnull().sum().sort_values(ascending=False)

## checking what kind of missing values are there in last two columns
main_df[main_df['last_evaluation'].isnull()]  ### Its NaN - not a number

### Filling the missing values with mean for numerical variables.
print(main_df['last_evaluation'].mean())
print(main_df['satisfaction_level'].mean())

### Since both of them are numerical we will fill with mean()
main_df.fillna(main_df['last_evaluation'].mean(),inplace=True)
main_df.fillna(main_df['satisfaction_level'].mean(),inplace=True)
main_df.last_evaluation.isnull().sum()
main_df.satisfaction_level.isnull().sum()

### Checking for specific observation
main_df[main_df['employee_id']==4150]

#OR by .loc
# main_df.loc[main_df['employee_id'] == 3794]

### Removing employee ID
## employee_id variable is not influencing our dataset in actul, it's just like employee name. Hence we will remove it.

main_df = main_df.drop(columns="employee_id")
main_df_final = main_df
main_df_final.head()

#### Groupby - we can group all the departments of company and see which department have done much projects or such kind of questions

# IN result we can say that it shows the work or tasks done by each department by group.
main_df_final.groupby(main_df_final.department).sum()
main_df_final.groupby(main_df_final.department).mean()
main_df_final['department'].value_counts()

### Checking how many employees have left and not.
main_df_final['left'].value_counts()
### Here,  0 11428 no - not left 
###        1 3571 yes - left

## -----------------------------------------------------------------------------------------------##

# **Step 5: Visualization** 

#### **Plotting correlation matrix**
# def plot_corr(df,size=10):
#   ## Function plots a graphical correlation matrix for each pair of columns in the dataframe.

#     Input:
#         df: pandas DataFrame
#         size: vertical and horizontal size of the plot

#     corr = df.corr()
#     fig, ax = plt.subplots(figsize=(size, size))
#     ax.legend()
#     cax = ax.matshow(corr)
#     fig.colorbar(cax)
#     plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
#     plt.yticks(range(len(corr.columns)), corr.columns)
    
# plot_corr(main_df_final)

#By plotting, we can see-
#If we consider 'left' variable, then 'satisfaction_level' is highly correlated with 'left' by which we can say that 
#satisfaction level makes the impact on ppl for leaving the job. 
#Work accidents - is also having slightly higher impact on the employees with the decision of leaving.
#Time spend company - also affected on leaving rate. 

## ----------------------------------------------------------------------------------------------##

# **Step 6: Preparing our dataset for Machine Learning.**
#Here department variable is categorical is having categories like sales, hr, IT. It's not in numeric, hence computer 
#will not be able to understand, hence we need to do encoding.

### Performing one hot encoding on categorical data
categorical = ['department','salary']
main_df_final = pd.get_dummies(main_df_final,columns=categorical,drop_first=True)
main_df_final.head()
### It will create diff column for each category in categorical variable
main_df_final.shape
## here we have 19 columns now.

### Checking how many employees in the dataset have left
len(main_df_final.loc[main_df_final['left']==1])
main_df_final.info()

## ----------------------------------------------------------------------------------------------##

# **Step 7: Splitting and standardizing dataset for ML**
from sklearn.model_selection import train_test_split

# We remove our label values from train data
X = main_df_final.drop(['left'],axis=1).values

# We assigned our label variable to test data
Y = main_df_final['left'].values

## Split X and Y into 70:30 ratio Train:Test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

## -----------------------------------------------------------------------------------------------##

## Step 8 : Feature Engineering

## Normalizing the data
# Why do we need to scale our data?
  # The data is measures in different units which is inconsistant. To make each variable to be equally weighted by a ML
  # classifier we scale our data.  
# We don't scale Y data because it includes label variable which will remain unaffected. That's why we only scale our Xdata
# by scaling data ..data become standard and data do not get skewed by any one data point.
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fit(raw_documents[, y]): Learn a vocabulary dictionary of all tokens in the raw documents.
# fit_transform(raw_documents[, y]): Learn the vocabulary dictionary and return term-document 
  # matrix. This is equivalent to fit followed by the transform, but more efficiently implemented.

X_train

df_train = pd.DataFrame(X_train)
df_train.head()

df_train.describe(include="all")

##-----------------------------------------------------------------------------------------------##

## Step 9 : Training LOGISTIC REGRESSION MODEL

from sklearn.linear_model import LogisticRegression

### creating a model
classifier_model = LogisticRegression()

### passing training data to model
classifier_model.fit(X_train,Y_train)

### predicting values x_test using model and storing the values in y_pred
y_pred = classifier_model.predict(X_test)

### interception and coefficient of model
print(classifier_model.intercept_)
print(classifier_model.coef_)

### printing values for better understanding
print(list(zip(Y_test, y_pred)))


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

### creating and printing confusion matrix
conf_matrix = confusion_matrix(Y_test,y_pred)
print(conf_matrix)

### Creating and printing classification report
print("Classification Report: ")
print(classification_report(Y_test,y_pred))


### Creating and printing accuracy score
acc = accuracy_score(Y_test,y_pred)
print("Accuracy {0:.2f}%".format(100*accuracy_score(y_pred, Y_test)))

## Result Analysis for Logistic classifier:
    
  #- After evaluating the Logistic Model as we can see in the output that we get 78.84% accuracy.
  #- Recall - Recall(TPR) means how much model predicting correctly. We got recall for 0 as 93%, which means
  #that model is predicting good for 0 i.e employees that don't left. But for 1 i.e employees that left it it 
  #predicting correctly only 35% Which is not good at all. And I Would never use such model.
#- Again if we check f-score then it is not giving good value of f-score

# So looking overall performance I rejected this model and try to build another classifier.
# Here we will built Random Forest. 

## ----------------------------------------------------------------------------------------------##

## Step 10 : Random Forest


from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(n_estimators=100,random_state=10) ## it will built 100 DT in background

#fit the model on the data and predict the values

random_forest_model.fit(X_train,Y_train)

y_pred_rf = random_forest_model.predict(X_test)


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

### creating and printing confusion matrix
conf_matrix_rf = confusion_matrix(Y_test,y_pred_rf)
print(conf_matrix_rf)

### Creating and printing classification report
print("Classification Report: ")
print(classification_report(Y_test,y_pred_rf))


### Creating and printing accuracy score
acc = accuracy_score(Y_test,y_pred_rf)
print("Accuracy {0:.2f}%".format(100*accuracy_score(y_pred_rf, Y_test)))

## Result Analysis for Random Forest classifier:
    
    #- After evaluating the Random Forest Model as we can see in the output that we get 98.84% accuracy.
    #- Recall - We are getting good recall score

#So looking overall performance We will keep random forest as our best predictor.    

### **Note: If we have been asked that what variable is most influential to make employee left. 
#             We can find this by '**feature_importances_**'

hr_data.head()

main_df_final.drop(['left'],axis=1).columns

import pandas as pd
feature_importances = pd.DataFrame(random_forest_model.feature_importances_,
                                   index = pd.DataFrame(X_train).columns,
                                    columns=['importance']).sort_values('importance',ascending=False)

feature_importances


#Here we can see that index of 5 variable is showing greater impact on the employee retention 
#following by index 0 and 2 and then 1 Index 5 variable is 'Satisfaction level' following by 
#'Number of projects' and then 'time spend in company'."""

## ==============================================================================================##
## ==============================================================================================##
## ==============================================================================================##


# **Deep Learning**

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=18))
model.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))

model.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

# Display Model Summary and Show Parameters
model.summary()

# Start Training Our Classifier 
batch_size = 10
epochs = 25

history = model.fit(X_train,
                    Y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# """One Epoch is when an ENTIRE dataset is passed forward and backward through the neural 
# network only ONCE."""

# Plotting our loss charts
# import matplotlib.pyplot as plt

# history_dict = history.history

# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# epochs = range(1, len(loss_values) + 1)

# line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
# line2 = plt.plot(epochs, loss_values, label='Training Loss')
# plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
# plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
# plt.xlabel('Epochs') 
# plt.ylabel('Loss')
# plt.grid(True)
# plt.legend()
# plt.show()
# - By looking at above plot we can see that loss is getting reduced in training and testing

# Plotting our accuracy charts
# import matplotlib.pyplot as plt

# history_dict = history.history

# print(history_dict)

# # acc_values = history_dict['accuracy']
# val_acc_values = history_dict['val_accuracy']
# epochs = range(1, len(loss_values) + 1)

# line1 = plt.plot(epochs, val_acc_values, label='Validation/Test Accuracy')
# line2 = plt.plot(epochs, acc_values, label='Training Accuracy')
# plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
# plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
# plt.xlabel('Epochs') 
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.legend()
# plt.show()

# By looking at above graph we can see that the accuracy is increasing with each epoch and there is no much difference between training predicted values and testing predicted values.
# We can clearly see from this graph that the accuracy rate is constantly increasing till the last point.
# But there is always a scope for improving the model.
# We will try to improve the model further."""

# **Displaying the Classification Report and Confusion Matrixt**

predictions = model.predict(X_test)
predictions = (predictions > 0.5)

print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

# **Let's a Deeper Model**

from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout

model2 = Sequential()

# Hidden Layer 1
model2.add(Dense(270, activation='relu', input_dim=18, kernel_regularizer=l2(0.01)))
model2.add(Dropout(0.3, noise_shape=None, seed=None))

# Hidden Layer 1
model2.add(Dense(180, activation='relu', input_dim=18, kernel_regularizer=l2(0.01)))
model2.add(Dropout(0.3, noise_shape=None, seed=None))

# Hidden Layer 2
model2.add(Dense(90, activation = 'relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3, noise_shape=None, seed=None))

model2.add(Dense(1, activation='sigmoid'))

model2.summary()


model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# **Training our Deeper Model**

batch_size = 10
epochs = 25

history = model2.fit(X_train,
                    Y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


predictions = model2.predict(X_test)
predictions = (predictions > 0.5)

print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


# 