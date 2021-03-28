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


#drop ticket no. it is useless
td=td.drop(['Ticket'],axis=1)

#Data Visualisation
sns.countplot(trd['Survived']) #around 60% died

sns.countplot(trd['Pclass'])
sns.countplot('Pclass',data=trd,hue='Survived') #survivality of 3rd class is very very very less

sns.countplot('Sex',data=trd,hue='Survived') #survivality of male is very less

#max age was 80
bins=[0,10,20,30,40,50,60,70,80]
group=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80']
td['age_bin']=pd.cut(td['Age'],bins,labels=group)
age_bin=pd.crosstab(td['age_bin'],td['Survived'])
age_bin.div(age_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.xlabel('Age Group')
plt.ylabel('Percent')

sns.countplot(x="age_bin", hue="Survived", data=td) #survivality of age 20-30 is very less , compared to others

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

td['Fare_bin'] = pd.cut(td['Fare'], bins=[0,7.90,14.45,31.28,120],labels=['Low','Mid','abv_avg','High'])
sns.countplot('Fare_bin',data=td,hue='Survived') #abv_avg and high fare prices gave more chances of surivavilty


sns.countplot('Embarked',data=td,hue='Survived')
# %OF survivality in C(57)>Q(33)=S(33) approx

#Feature Engineering

# extracting the surname from the names to help predict other features
td['Title'] = td['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
td['Title'].value_counts()














