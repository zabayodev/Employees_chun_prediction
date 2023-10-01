# Employees_chun_prediction
This is the repository for showcasing the steps needed to predict the employees Chun in telecom sector to know the employees who will leave and those who will stay.

## Chosing the programming language
The first step in the precting the employee chun is making the choice of a programming language between python programming and the R programming as well as scala. In this project we will use the Python programming as it is the most familiar with data science space.

## Research on the data with employee chun data
The starting point is to find the data that are specific for telecom companies which have the dataset for predicting customer chun for the employees who left and those who stayed with their specific causes within the dataset.

## Importing Python libraries for manipulating data and building model
There are the specific python libraries for building machine learning models such as pandas, numpy, scikit-learn, seaborn, smote for workig with embalanced dataset and moch more.

import pandas as pd  
import nmpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  

## Import data
 The obtained data are imported in python using different python libraries such as read_csv, read_excel, read_sql and much more as the libraries depends on the data format.

 df = pd.read_csv("Add dataset path")

 ## Perform Data Exploratory Analysis(EDA)
 The EDA is used to perform data preprocessing by looking into the data in deep to understand the obtained information such as the missing data, dropping the unwanted columns, normalizing the columns names and data visualization.

 df = df.drop(["Adding the columns to drop"])
df = df.shape
df = df.dtypes
df = df.isnull().sum()
df = df.dropna()
df = sns.hist("Adding columns for histogram visualization")

## Data Modeling
Data modeling is mostly turning the non numerical columns into numerical columns such as gender.

0 for Female
1 for Male

## Removing the imbalance data
Turning the imbalanced data into balanced data using the dammies function.

df1 = pd.get_dummies(data=df1, columns=[])

## Importing the scikit_learn
For building machine learning model, the machine learning libraries is imported for fitting the data.

from sklearn.model_selection import train_test_split

## Splitting the data into the test and training
the dataset is splited into two to separate the test and the train in building macine learning model.

X_train, X_test, y_train, y_test = train_test_split(" Definind the split method")

## Importing the required models to be used in predicting the ouput 
Different models are imported within the dataset to be able to predict the oucomes.

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

## Selecting the best model score
The selection of the models is based on the chosen models and thier perfomance

reg=XGBClassifier()
reg.fit(X_train, y_train)

## Verfication of the model using the test dataset
The test dataset is used to verify the performance of th models and generalize the conclusion

y_test["Adding the test subject"]

## Importing the confusion matrix to elaborate the prediction
The confusion matrix is used to verify the true positive, true negative, false negative, false positive.

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)

## Importing the classification report
This is used to find the metrics in the building of the model and their conclusion

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
