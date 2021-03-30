# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the train and test datasets
trd=pd.read_csv("train.csv")
tsd=pd.read_csv("test.csv")

trd.describe()
td = pd.concat([trd, tsd], ignore_index=True, sort  = False)

#Exploratory Data Analysis

trd.isnull().sum()
tsd.isnull().sum()
td.isnull().sum()

#drop ticket no. it is useless
td=td.drop(['Ticket'],axis=1)

#Data Visualisation
sns.countplot(trd['Survived']) #around 60% died

sns.countplot(trd['Pclass'])
sns.countplot('Pclass',data=trd,hue='Survived') #survivality of 3rd class is very very very less

sns.countplot('Sex',data=trd,hue='Survived') #survivality of male is very less


#max SibSp=8,min=0
bins=[0,1,2,3,4,5,6,7,8]
group=['0-1','1-2','2-3','3-4','4-5','5-6','6-7','7-8']
td['SibSp_bin']=pd.cut(td['SibSp'],bins,labels=group)
sns.countplot('SibSp_bin',data=td,hue='Survived') #survivality decreases drastically if travelled with more than 2spouse or sibling

#max parch=6,min=0
bins=[0,1,2,3,4,5,6]
group=['0-1','1-2','2-3','3-4','4-5','5-6']
td['Parch_bin']=pd.cut(td['Parch'],bins,labels=group)
sns.countplot('Parch_bin',data=td,hue='Survived') #survivality decreases drastically if travelled with more than 2 children

td['family']=td['Parch']+td['SibSp']
td['alone']=td['family']==0 
#travelling alone had high chance of survivality

td=td.drop(['Parch','SibSp','family'],axis=1)

td['Fare_bin'] = pd.cut(td['Fare'], bins=[0,7.90,14.45,31.28,120],labels=['Low','Mid','abv_avg','High'])
sns.countplot('Fare_bin',data=td,hue='Survived') #abv_avg and high fare prices gave more chances of surivavilty


sns.countplot('Embarked',data=td,hue='Survived')
# % of survivality in C(57)>Q(33)=S(33) approx

#Feature Engineering

# extracting the surname from the names to help predict other features
td['Title'] = td['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
td['Title'].value_counts()


title_map = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 4,"Countess": 4,
                 "Ms": 4, "Lady": 4, "Jonkheer": 4, "Don": 4, "Dona" : 4, "Mme": 4,"Capt": 4,"Sir": 4 }

td['Title'] = td['Title'].map(title_map)

sns.countplot('Title',data=td,hue='Survived') #survivality order Mrs>Miss>Master>Mr

# now we do not require the name col , so we can drop it.
td=td.drop(['Name'],axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cols=['Sex',['alone']]
for col in cols:
    td[col]=le.fit_transform(td[col])

#filling the age 
# We use the median age for each title (Mr, Mrs, Miss,Master, Others)
td["Age"].fillna(td.groupby("Title")["Age"].transform("median"), inplace=True)
td.isnull().sum()

#max age was 80
bins=[0,10,20,30,40,50,60,70,80]
group=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80']
td['age_bin']=pd.cut(td['Age'],bins,labels=group)
age_bin=pd.crosstab(td['age_bin'],td['Survived'])
age_bin.div(age_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)


sns.countplot(x="age_bin", hue="Survived", data=td) #survivality of age 20-30 is very less , compared to others

#Embarked
Class1=td[td['Pclass']==1]['Embarked'].value_counts()
Class2=td[td['Pclass']==2]['Embarked'].value_counts()
Class3=td[td['Pclass']==3]['Embarked'].value_counts()
df=pd.DataFrame([Class1,Class2,Class3])
df.index=['1stClass','2ndClass','3rdClass']
df.plot(kind='bar', stacked=True)
#conclusion- in all 3 classes we have more that 50% as S Embarked

#filling Embarked
td['Embarked']=td['Embarked'].fillna('S')

#Encoding embarked
td['Embarked']=le.fit_transform(td['Embarked'])

# fill missing Fare with median fare for each Pclass
td["Fare"].fillna(td.groupby("Pclass")["Fare"].transform("median"), inplace=True)

#dropping cabin as it has a lot of missing values
td=td.drop(['Cabin','SibSp_bin','Parch_bin','Fare_bin','age_bin'],axis=1)

#Making an Age Class 

td['Age_bin'] = pd.cut(td['Age'], 5)
td[['Age_bin', 'Survived']].groupby(['Age_bin'], as_index=False).mean().sort_values(by='Age_bin', ascending=True)
td.loc[ td['Age'] <= 16, 'Age'] = 0
td.loc[(td['Age'] > 16) & (td['Age'] <= 32), 'Age'] = 1
td.loc[(td['Age'] > 32) & (td['Age'] <= 48), 'Age'] = 2
td.loc[(td['Age'] > 48) & (td['Age'] <= 64), 'Age'] = 3
td.loc[ td['Age'] > 64, 'Age']=4

#Making a Fare Class
td['Fare_bin'] = pd.qcut(td['Fare'], 5)
td[['Fare_bin', 'Survived']].groupby(['Fare_bin'], as_index=False).mean().sort_values(by='Fare_bin', ascending=True)

td.loc[ td['Fare'] <= 7.854, 'Fare'] = 0
td.loc[(td['Fare'] > 7.854) & (td['Fare'] <= 10.5), 'Fare'] = 1
td.loc[(td['Fare'] >10.5) & (td['Fare'] <=  21.558), 'Fare']   = 2
td.loc[(td['Fare'] > 21.558) & (td['Fare'] <=  41.579), 'Fare']   = 3
td.loc[ td['Fare'] > 41.579, 'Fare'] = 4

td=td.drop(['Fare_bin','Age_bin'],axis=1)
td.isnull().sum()

#drop passengerId
td=td.drop(['PassengerId'],axis=1)
#splitting x_train and x_test
x_test = td[td.Survived.isnull()]
x_test = x_test.drop(['Survived'], axis = 1)

x_train=td.dropna()
y_train = td['Survived'].dropna()
x_train= x_train.drop(['Survived'], axis = 1)

#splitting x_train into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(x_train,y_train,test_size=0.25 , random_state=42) 

#Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

x_train.isnull().sum()
y_train.isnull().sum()
x_test.isnull().sum()

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
# from logistic regression i got 79.80%



# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
# from SVM i got 78.75%


# Training the Decision Tree model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
# from Decision Tree i got 77.99%%



# Training the Random Forest model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
# from Random Forest i got 80.55% WINNER


#Predicting the actual test set
y_pred = classifier.predict(x_test)

#creating submission file
og_test_set=pd.read_csv("test.csv")
sol=pd.read_csv("submit.csv")
sol["Survived"]=y_pred
sol["PassengerId"]=og_test_set["PassengerId"]

#converting to csv file
pd.DataFrame(sol,columns=["PassengerId","Survived"]).to_csv("final_answer.csv")

# Result - 77.03%






